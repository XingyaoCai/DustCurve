#!/usr/bin/env python3
"""
MPI-parallelized version of catalog processing code.
Run with: mpirun -n <num_processes> python this_script.py
"""

from mpi4py import MPI
import os
import astropy.units
from scipy.fftpack import hilbert
from tqdm import tqdm
import pickle
import FunctionLib as FL

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def process_catalog_chunk(catalog_items, rank_id):
    """
    Process a chunk of catalog items on a single MPI process.

    Args:
        catalog_items: List of (id, catalog) tuples to process
        rank_id: MPI rank for progress tracking

    Returns:
        List of processed (id, catalog) tuples
    """
    processed_items = []

    # Use tqdm only on rank 0 to avoid cluttered output
    iterator = tqdm(
        catalog_items, desc=f"Rank {rank_id}") if rank_id == 0 else catalog_items

    for id, catalog in iterator:
        # Set default values
        catalog['properties']['Sample_Flag'] = True
        catalog['properties']['Sample_Reason'] = 'Spectrum Available'

        # Check for prism filepath
        if catalog.get('prism_filepath') is None:
            catalog['properties']['Sample_Flag'] = False
            catalog['properties']['Sample_Reason'] = 'No Prism Spectrum'
            processed_items.append((id, catalog))
            continue

        # Check for grating filepaths
        if catalog.get('grating_filepaths') == {}:
            catalog['properties']['Sample_Flag'] = False
            catalog['properties']['Sample_Reason'] = 'No Grating Spectrum'
            processed_items.append((id, catalog))
            continue

        try:
            # Load and process spectrum
            prism_spectrum = FL.Load_Spectrum_From_Fits(
                catalog['prism_filepath'],
                catalog['determined_redshift']
            )

            inf, sup = prism_spectrum.dual_boundarys()

            if inf is None or sup is None:
                catalog['properties']['Sample_Flag'] = False
                catalog['properties']['Sample_Reason'] = 'Prism Spectrum Boundarys Not Found'
                processed_items.append((id, catalog))
                continue

            if inf > 125 * astropy.units.nm or sup < 258 * astropy.units.nm:
                catalog['properties']['Sample_Flag'] = False
                catalog['properties']['Sample_Reason'] = 'Prism Spectrum Boundarys Not In UV Range'
                processed_items.append((id, catalog))
                continue

            # Process grating spectra
            halpha_coverage = False
            hbeta_coverage = False

            for filter,grating_filepath in catalog['grating_filepaths'].items():
                grating_spectrum = FL.Load_Spectrum_From_Fits(
                    grating_filepath,
                    catalog['determined_redshift']
                )

                if grating_spectrum is None:
                    continue

                # Check for H-alpha coverage
                inf, sup = grating_spectrum.dual_boundarys()
                if inf is None or sup is None:
                    catalog['properties']['Sample_Flag'] = False
                    catalog['properties']['Sample_Reason'] = 'Grating Spectrum Boundarys Not Found'
                    processed_items.append((id, catalog))
                    continue

                if inf <= 656.3 * astropy.units.nm <= sup:
                    halpha_coverage = True
                if inf <= 486.1 * astropy.units.nm <= sup:
                    hbeta_coverage = True

            if not halpha_coverage and not hbeta_coverage:
                catalog['properties']['Sample_Flag'] = False
                catalog['properties']['Sample_Reason'] = 'Grating no H-alpha and H-beta Coverage'
                processed_items.append((id, catalog))
                continue

            if halpha_coverage and not hbeta_coverage:
                catalog['properties']['Sample_Flag'] = False
                catalog['properties']['Sample_Reason'] = 'Grating no H-beta Coverage'
                processed_items.append((id, catalog))
                continue

            if not halpha_coverage and hbeta_coverage:
                catalog['properties']['Sample_Flag'] = False
                catalog['properties']['Sample_Reason'] = 'Grating no H-alpha Coverage'
                processed_items.append((id, catalog))
                continue

        except Exception as e:
            # Handle any errors in spectrum processing
            catalog['properties']['Sample_Flag'] = False
            catalog['properties']['Sample_Reason'] = f'Error processing spectrum: {str(e)}'
            processed_items.append((id, catalog))
            continue

        # If we get here, the catalog passed all checks
        processed_items.append((id, catalog))

    return processed_items


def main():
    """Main MPI processing function."""
    DJAv4Catalog = FL.Spectrum_Catalog()
    DJAv4Catalog.load_from_pkl(os.path.expanduser(
        '~/DustCurve/DJAv4Catalog.pkl'))

    if rank == 0:
        print(f"Starting MPI processing with {size} processes...")

        # Collect all catalog items on rank 0
        all_catalog_items = []
        for id, catalog in DJAv4Catalog.catalog_iterator():
            all_catalog_items.append((id, catalog))

        print(f"Total catalog items to process: {len(all_catalog_items)}")

        # Divide work among processes
        chunk_size = len(all_catalog_items) // size
        remainder = len(all_catalog_items) % size

        chunks = []
        start_idx = 0

        for i in range(size):
            # Distribute remainder among first few processes
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            chunks.append(all_catalog_items[start_idx:end_idx])
            start_idx = end_idx

        print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    else:
        chunks = None

    # Scatter the chunks to all processes
    local_chunk = comm.scatter(chunks, root=0)

    if rank == 0:
        print("Processing chunks in parallel...")

    # Process local chunk
    processed_chunk = process_catalog_chunk(local_chunk, rank)

    # Gather all processed results back to rank 0
    all_processed = comm.gather(processed_chunk, root=0)

    if rank == 0:
        print("Gathering results and updating catalog...")

        # Flatten the list of processed chunks
        all_items = []
        for chunk in all_processed:
            all_items.extend(chunk)

        # Update the catalog with processed items
        # Note: This assumes DJAv4Catalog has a method to update from processed items
        # You may need to adapt this part based on your catalog structure
        for id, catalog in all_items:
            DJAv4Catalog.update_catalog_item(id, catalog)

        # Save the updated catalog
        output_path = os.path.expanduser('~/DustCurve/DJAv4Catalog.pkl')
        DJAv4Catalog.save_catalog_to_pkl(output_path)

        print(f"Processing complete! Results saved to {output_path}")

        # Print summary statistics
        total_items = len(all_items)
        flagged_true = sum(
            1 for _, cat in all_items if cat['properties']['Sample_Flag'] == True)
        flagged_false = total_items - flagged_true

        print(f"Summary:")
        print(f"  Total items processed: {total_items}")
        print(
            f"  Sample_Flag = True: {flagged_true} ({flagged_true/total_items*100:.1f}%)")
        print(
            f"  Sample_Flag = False: {flagged_false} ({flagged_false/total_items*100:.1f}%)")


if __name__ == "__main__":
    main()
