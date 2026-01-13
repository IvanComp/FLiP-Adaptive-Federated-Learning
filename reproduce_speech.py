import torch
import sys
import os

# Add Docker directory to path
sys.path.append(os.path.abspath("Docker"))

try:
    import taskA
    print("Successfully imported taskA")
except ImportError as e:
    print(f"Failed to import taskA: {e}")
    sys.exit(1)

def verify_m5_model():
    print("Verifying M5 model...")
    try:
        model = taskA.get_dynamic_model(num_classes=35, model_name="m5")
        print(f"Successfully created M5 model: {model}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 8000) # (Batch, Channel, Length)
        output = model(dummy_input)
        print(f"Forward pass output shape: {output.shape}")
        
        assert output.shape == (2, 35), f"Expected output shape (2, 35), got {output.shape}"
        print("Forward pass successful")
    except Exception as e:
        print(f"M5 model verification failed: {e}")
        raise e

def verify_dataset_config():
    print("Verifying SPEECHCOMMANDS dataset config...")
    if "SPEECHCOMMANDS" in taskA.AVAILABLE_DATASETS:
        print("SPEECHCOMMANDS found in AVAILABLE_DATASETS")
    else:
        print("SPEECHCOMMANDS NOT found in AVAILABLE_DATASETS")
        sys.exit(1)

if __name__ == "__main__":
    verify_dataset_config()
    verify_m5_model()
    print("All verifications passed!")
