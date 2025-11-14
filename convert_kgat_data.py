#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert KGAT dataset format to FedKGAT2 format
"""
import os
import pandas as pd
from pathlib import Path

def convert_last_fm_to_music():
    """
    Convert Last-FM dataset from KGAT format to music dataset format for FedKGAT2
    """
    # Paths
    kgat_data_dir = Path("/tmp/kgat_repo/Data/last-fm")
    output_dir = Path("./data/music")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Converting Last-FM dataset to music format...")
    print("=" * 60)
    
    # 1. Convert item_list.txt to item_index2entity_id.txt
    print("\n[1/3] Processing item_index2entity_id.txt...")
    # KGAT format: org_id remap_id freebase_id (with header)
    # Target format: item_id\tentity_id (both should be numeric IDs)
    item_df = pd.read_csv(
        kgat_data_dir / "item_list.txt",
        sep='\s+',
        names=['org_id', 'remap_id', 'freebase_id'],
        skiprows=1  # Skip header
    )
    
    # Load entity_list to map freebase_id to entity remap_id
    # Format varies: some have header "org_id remap_id", some don't
    # Some lines may have more than 2 fields, so we parse manually
    entity_file = kgat_data_dir / "entity_list.txt"
    entity_list = []
    with open(entity_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if i == 0:
                # Check if first line is a header
                if parts[0].startswith('org_id') or parts[0].startswith('freebase_id'):
                    continue  # Skip header
            if len(parts) >= 2:
                # Take only first two fields (freebase_id and remap_id)
                try:
                    entity_list.append({
                        'freebase_id': parts[0],
                        'entity_remap_id': int(parts[1])
                    })
                except (ValueError, IndexError):
                    # Skip malformed lines
                    continue
    
    entity_df = pd.DataFrame(entity_list)
    
    # Create mapping: freebase_id -> entity_remap_id
    entity_dict = dict(zip(entity_df['freebase_id'], entity_df['entity_remap_id']))
    
    # Map item remap_id to entity remap_id via freebase_id
    item_mapping = []
    for _, row in item_df.iterrows():
        item_remap_id = row['remap_id']
        freebase_id = row['freebase_id']
        # Find corresponding entity remap_id
        if freebase_id in entity_dict:
            entity_remap_id = entity_dict[freebase_id]
            item_mapping.append({
                'item': item_remap_id,
                'id': entity_remap_id
            })
        else:
            # If entity not found, use item remap_id itself
            item_mapping.append({
                'item': item_remap_id,
                'id': item_remap_id
            })
    
    item_mapping_df = pd.DataFrame(item_mapping)
    item_mapping_df.to_csv(
        output_dir / "item_index2entity_id.txt",
        sep='\t',
        index=False,
        header=False
    )
    print(f"  ✓ Created item_index2entity_id.txt with {len(item_mapping_df)} items")
    
    # 2. Convert train.txt and test.txt to ratings_final.txt
    print("\n[2/3] Processing ratings_final.txt...")
    ratings_list = []
    
    # Process train.txt
    with open(kgat_data_dir / "train.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            item_ids = [int(x) for x in parts[1:]]
            # Create positive ratings (threshold > 0, so we use 1)
            for item_id in item_ids:
                ratings_list.append({
                    'userID': user_id,
                    'itemID': item_id,
                    'rating': 1
                })
    
    # Process test.txt
    with open(kgat_data_dir / "test.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            item_ids = [int(x) for x in parts[1:]]
            # Create positive ratings
            for item_id in item_ids:
                ratings_list.append({
                    'userID': user_id,
                    'itemID': item_id,
                    'rating': 1
                })
    
    ratings_df = pd.DataFrame(ratings_list)
    ratings_df.to_csv(
        output_dir / "ratings_final.txt",
        sep='\t',
        index=False,
        header=False
    )
    print(f"  ✓ Created ratings_final.txt with {len(ratings_df)} ratings")
    print(f"    - Users: {ratings_df['userID'].nunique()}")
    print(f"    - Items: {ratings_df['itemID'].nunique()}")
    
    # 3. Copy/convert kg_final.txt
    print("\n[3/3] Processing kg_final.txt...")
    # KGAT format: head_entity relation tail_entity (space-separated)
    # Target format: head_entity\trelation\ttail_entity (tab-separated)
    kg_df = pd.read_csv(
        kgat_data_dir / "kg_final.txt",
        sep='\s+',
        names=['head', 'relation', 'tail'],
        header=None
    )
    kg_df.to_csv(
        output_dir / "kg_final.txt",
        sep='\t',
        index=False,
        header=False
    )
    print(f"  ✓ Created kg_final.txt with {len(kg_df)} triplets")
    
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)


def convert_amazon_book():
    """
    Convert Amazon-book dataset from KGAT format to book dataset format for FedKGAT2
    """
    # Paths
    kgat_data_dir = Path("/tmp/kgat_repo/Data/amazon-book")
    output_dir = Path("./data/book")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Converting Amazon-book dataset to book format...")
    print("=" * 60)
    
    # 1. Convert item_list.txt to item_index2entity_id_rehashed.txt
    print("\n[1/3] Processing item_index2entity_id_rehashed.txt...")
    item_df = pd.read_csv(
        kgat_data_dir / "item_list.txt",
        sep='\s+',
        names=['org_id', 'remap_id', 'freebase_id'],
        skiprows=1
    )
    
    # Load entity_list to map freebase_id to entity remap_id
    # Format varies: some have header "org_id remap_id", some don't
    # Some lines may have more than 2 fields, so we parse manually
    entity_file = kgat_data_dir / "entity_list.txt"
    entity_list = []
    with open(entity_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if i == 0:
                # Check if first line is a header
                if parts[0].startswith('org_id') or parts[0].startswith('freebase_id'):
                    continue  # Skip header
            if len(parts) >= 2:
                # Take only first two fields (freebase_id and remap_id)
                try:
                    entity_list.append({
                        'freebase_id': parts[0],
                        'entity_remap_id': int(parts[1])
                    })
                except (ValueError, IndexError):
                    # Skip malformed lines
                    continue
    
    entity_df = pd.DataFrame(entity_list)
    
    # Create mapping: freebase_id -> entity_remap_id
    entity_dict = dict(zip(entity_df['freebase_id'], entity_df['entity_remap_id']))
    
    # Map item remap_id to entity remap_id via freebase_id
    item_mapping = []
    for _, row in item_df.iterrows():
        item_remap_id = row['remap_id']
        freebase_id = row['freebase_id']
        # Find corresponding entity remap_id
        if freebase_id in entity_dict:
            entity_remap_id = entity_dict[freebase_id]
            item_mapping.append({
                'item': item_remap_id,
                'id': entity_remap_id
            })
        else:
            # If entity not found, use item remap_id itself
            item_mapping.append({
                'item': item_remap_id,
                'id': item_remap_id
            })
    
    item_mapping_df = pd.DataFrame(item_mapping)
    item_mapping_df.to_csv(
        output_dir / "item_index2entity_id_rehashed.txt",
        sep='\t',
        index=False,
        header=False
    )
    print(f"  ✓ Created item_index2entity_id_rehashed.txt with {len(item_mapping_df)} items")
    
    # 2. Convert train.txt and test.txt to BX-Book-Ratings.csv
    print("\n[2/3] Processing BX-Book-Ratings.csv...")
    ratings_list = []
    
    # Process train.txt
    with open(kgat_data_dir / "train.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            item_ids = [int(x) for x in parts[1:]]
            for item_id in item_ids:
                ratings_list.append({
                    'userID': user_id,
                    'itemID': item_id,
                    'rating': 1
                })
    
    # Process test.txt
    with open(kgat_data_dir / "test.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            item_ids = [int(x) for x in parts[1:]]
            for item_id in item_ids:
                ratings_list.append({
                    'userID': user_id,
                    'itemID': item_id,
                    'rating': 1
                })
    
    ratings_df = pd.DataFrame(ratings_list)
    # Note: Book dataset uses semicolon separator according to the guide
    ratings_df.to_csv(
        output_dir / "BX-Book-Ratings.csv",
        sep=';',
        index=False,
        header=False
    )
    print(f"  ✓ Created BX-Book-Ratings.csv with {len(ratings_df)} ratings")
    print(f"    - Users: {ratings_df['userID'].nunique()}")
    print(f"    - Items: {ratings_df['itemID'].nunique()}")
    
    # 3. Convert kg_final.txt to kg_rehashed.txt
    print("\n[3/3] Processing kg_rehashed.txt...")
    # Check if kg_final.txt exists (or is zipped)
    kg_file = None
    if (kgat_data_dir / "kg_final.txt").exists():
        kg_file = kgat_data_dir / "kg_final.txt"
    elif (kgat_data_dir / "kg_final.txt.zip").exists():
        # Extract zip file if needed
        print("  Extracting kg_final.txt.zip...")
        import zipfile
        with zipfile.ZipFile(kgat_data_dir / "kg_final.txt.zip", 'r') as zip_ref:
            zip_ref.extractall(kgat_data_dir)
        kg_file = kgat_data_dir / "kg_final.txt"
    
    if kg_file and kg_file.exists():
        kg_df = pd.read_csv(
            kg_file,
            sep='\s+',
            names=['head', 'relation', 'tail'],
            header=None
        )
    else:
        print("  ⚠ Warning: kg_final.txt not found, creating empty kg_rehashed.txt")
        kg_df = pd.DataFrame(columns=['head', 'relation', 'tail'])
    
    kg_df.to_csv(
        output_dir / "kg_rehashed.txt",
        sep='\t',
        index=False,
        header=False
    )
    print(f"  ✓ Created kg_rehashed.txt with {len(kg_df)} triplets")
    
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    # Convert Last-FM to music format
    convert_last_fm_to_music()
    
    # Convert Amazon-book to book format  
    convert_amazon_book()
    
    print("\n" + "=" * 60)
    print("All datasets converted!")
    print("=" * 60)

