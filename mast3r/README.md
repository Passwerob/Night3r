# MASt3R Module

This directory contains the MASt3R (Matching Anything by Segmenting with 3D Representations) codebase.

## Source

**This code was copied from the MASt3R repository.** 

This is a local copy of the MASt3R codebase included in Dark3R to ensure compatibility and avoid external dependency issues. Please refer to the original MASt3R repository for:
- Full documentation
- Latest updates
- Issue tracking
- License information

## Usage in Dark3R

Dark3R uses MASt3R as the base model for 3D reconstruction. The model weights are loaded from HuggingFace, and this local copy provides the necessary utilities and core functionality for:
- Sparse global alignment
- Image pair generation
- Fast nearest neighbor matching
- Retrieval-based scene graph construction
- Loss functions and optimization

## Note

This directory should be treated as read-only for Dark3R purposes. Any modifications to MASt3R code should be made in the original repository and then copied here if needed.

