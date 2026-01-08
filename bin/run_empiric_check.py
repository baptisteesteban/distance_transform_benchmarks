#!/usr/bin/env python3
"""
Run empiric_check on all images in the ECSSD dataset.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run empiric_check on ECSSD dataset images")
    parser.add_argument("-n", "--num-images", type=int, default=None,
                        help="Number of images to process (default: all)")
    args = parser.parse_args()

    gray_dir = Path("/data/ecssd/gray_png")
    mask_dir = Path("/data/ecssd/masks")
    empiric_check_bin = Path("/workspaces/dist_benchmarks/build/Debug/bin/empiric_check")
    
    # Change me
    lambda_value = 1
    
    if not empiric_check_bin.exists():
        print(f"Error: empiric_check binary not found at {empiric_check_bin}", file=sys.stderr)
        print("Please build the project first.", file=sys.stderr)
        sys.exit(1)
    
    if not gray_dir.exists():
        print(f"Error: Gray image directory not found at {gray_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not mask_dir.exists():
        print(f"Error: Mask directory not found at {mask_dir}", file=sys.stderr)
        sys.exit(1)
    
    gray_images = sorted(gray_dir.glob("*.png"))
    
    if not gray_images:
        print(f"Error: No PNG files found in {gray_dir}", file=sys.stderr)
        sys.exit(1)
    
    if args.num_images is not None:
        gray_images = gray_images[:args.num_images]
    
    print(f"Found {len(gray_images)} images to process")
    print(f"Using lambda value: {lambda_value}")
    print(f"Empiric check binary: {empiric_check_bin}")
    print("-" * 80)
    
    passed = 0
    failed = 0
    failed_images = []
    
    for i, gray_image in enumerate(gray_images):
        mask_image = mask_dir / gray_image.name
        
        if not mask_image.exists():
            print(f"SKIP: {gray_image.name} - No corresponding mask found")
            continue

        cmd = [
            str(empiric_check_bin),
            str(gray_image),
            str(lambda_value),
            str(mask_image)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✓ PASS: {gray_image.name}")
                passed += 1
            else:
                print(f"✗ FAIL: {gray_image.name}")
                if result.stderr:
                    print(f"  Error: {result.stderr.strip()}")
                failed += 1
                failed_images.append(gray_image.name)
                
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT: {gray_image.name}")
            failed += 1
            failed_images.append(gray_image.name)
        except Exception as e:
            print(f"✗ ERROR: {gray_image.name} - {e}")
            failed += 1
            failed_images.append(gray_image.name)
    
    print("-" * 80)
    print(f"\nSummary:")
    print(f"  Total images: {len(gray_images)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if failed_images:
        print(f"\nFailed images:")
        for img in failed_images:
            print(f"  - {img}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
