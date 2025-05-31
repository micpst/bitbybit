# Optimized hash length configurations for improved efficiency scores

# Strategy 1: Layer-Type Based (Simple & Effective)
resnet20_layertype_config = {
    # --- Top-level Conv (early layer, high redundancy) ---
    "conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },

    # --- Layer 1 convs (16 channels) ---
    "layer1.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },

    # --- Layer 2 convs (32 channels) ---
    "layer2.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.0.downsample.0": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },

    # --- Layer 3 convs (64 channels) ---
    "layer3.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.0.downsample.0": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },

    # --- Fully Connected Layer - Critical for classification ---
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    }
}

# Strategy 2: Depth-Based Progressive Scaling
resnet20_depth_config = {
    # Early layers (depth 0-25%)
    "conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },

    # Mid-early layers (depth 25-50%)
    "layer1.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.0.downsample.0": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },

    # Mid-late layers (depth 50-75%)
    "layer2.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer2.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer2.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer2.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },

    # Deep layers (depth 75-100%)
    "layer3.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer3.0.downsample.0": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer3.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer3.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer3.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer3.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },

    # Final layer - Maximum precision
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    }
}

# Strategy 3: Channel-Count Proportional
resnet20_channel_config = {
    # 3->16 channels
    "conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 128
    },

    # 16->16 channels (small)
    "layer1.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256
    },

    # 16->32 and 32->32 channels (medium)
    "layer2.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.0.downsample.0": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },
    "layer2.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512
    },

    # 32->64 and 64->64 channels (large)
    "layer3.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.0.downsample.0": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },

    # 64->10 final classification
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    }
}

# Strategy 4: Ultra-Aggressive (Maximum Efficiency)
resnet20_aggressive_config = {
    # All conv layers use minimal hashes
    "conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 128
    },
    
    # Layer 1 - very small
    "layer1.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 128
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 128
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 128
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 128
    },
    "layer1.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 128
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 128
    },

    # Layer 2 - small
    "layer2.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 256
    },
    "layer2.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 256
    },
    "layer2.0.downsample.0": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 256
    },
    "layer2.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 256
    },
    "layer2.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 256
    },
    "layer2.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 256
    },
    "layer2.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 256
    },

    # Layer 3 - medium
    "layer3.0.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 512
    },
    "layer3.0.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 512
    },
    "layer3.0.downsample.0": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 512
    },
    "layer3.1.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 512
    },
    "layer3.1.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 512
    },
    "layer3.2.conv1": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 512
    },
    "layer3.2.conv2": {
        "hash_kernel_type": "random_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 512
    },

    # Only FC layer gets decent hash length
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 128,
        "output_tile_size": 128,
        "hash_length": 1024
    }
}