#!/bin/bash
# Copy trained models from runs/train/ to models/trained/

echo "Copying trained models..."

# Create target directory
mkdir -p models/trained

# Copy best.pt for each object
for obj_dir in runs/train/*/; do
    if [ -f "$obj_dir/weights/best.pt" ]; then
        obj_name=$(basename "$obj_dir")
        echo "  Copying $obj_name..."
        cp "$obj_dir/weights/best.pt" "models/trained/$obj_name.pt"
    fi
done

echo ""
echo "✓ Done! Trained models copied to models/trained/"
ls -lh models/trained/

