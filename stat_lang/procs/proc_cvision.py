"""PROC CVISION — Computer Vision (classification + object detection).

Modes
-----
classify          Classify images using pretrained ImageNet models.
generate_samples  Generate synthetic object-detection training data
                  (coloured shapes on random backgrounds).
train_detect      Fine-tune Faster R-CNN on labelled bounding-box data.
detect / score    Run object detection (pretrained or from model store).
serve             Alias for detect — loads a saved model and scores images.

Usage examples
--------------
    /* Generate synthetic training images */
    PROC CVISION mode=generate_samples out=annotations
         n_train=30 n_test=10 img_size=128 seed=42
         outdir='./work/images';
    RUN;

    /* Fine-tune Faster R-CNN */
    PROC CVISION DATA=train_annot mode=train_detect
         model_name=shape_detector epochs=5 lr=0.005;
        IMAGE image_path;
    RUN;

    /* Score new images with the trained model */
    PROC CVISION DATA=test_images mode=serve out=detections
         model_name=shape_detector confidence=0.5;
        IMAGE image_path;
    RUN;
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..parser.proc_parser import ProcStatement

try:
    import torch
    from PIL import Image, ImageDraw
    from torchvision import models, transforms

    HAS_VISION = True
except ImportError:
    HAS_VISION = False

try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    HAS_DETECTION = True
except ImportError:
    HAS_DETECTION = False

# Colour palette for synthetic shapes
_SHAPE_PALETTE: Dict[str, Tuple[str, str]] = {
    'circle': ('red', 'darkred'),
    'rectangle': ('blue', 'darkblue'),
    'triangle': ('green', 'darkgreen'),
}


class ProcCVision:
    """Computer Vision PROC — classification and object detection."""

    # ==================================================================
    # Entry point
    # ==================================================================
    def execute(
        self, data: pd.DataFrame, proc_info: ProcStatement, **kw
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}

        if not HAS_VISION:
            results['output_text'].append(
                "ERROR: torchvision or Pillow not installed"
            )
            return results

        mode = str(proc_info.options.get('mode', 'classify')).lower()

        if mode == 'classify':
            return self._classify(data, proc_info)
        elif mode == 'generate_samples':
            return self._generate_samples(proc_info, **kw)
        elif mode == 'train_detect':
            return self._train_detect(data, proc_info, **kw)
        elif mode in ('detect', 'score', 'serve'):
            return self._detect(data, proc_info, **kw)
        else:
            results['output_text'].append(f"ERROR: Unknown CVISION mode '{mode}'")
            return results

    # ==================================================================
    # MODE: classify  (original functionality)
    # ==================================================================
    def _classify(
        self, data: pd.DataFrame, proc_info: ProcStatement
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}

        image_col = str(proc_info.options.get('image', ''))
        model_name = str(
            proc_info.options.get('model_name', 'resnet18')
        ).lower()

        if not image_col:
            str_cols = data.select_dtypes(include='object').columns.tolist()
            if str_cols:
                image_col = str_cols[0]
            else:
                results['output_text'].append("ERROR: IMAGE column required")
                return results

        if image_col not in data.columns:
            results['output_text'].append(
                f"ERROR: Column '{image_col}' not found"
            )
            return results

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        model.eval()

        predictions: List[Dict] = []
        image_paths = data[image_col].dropna().astype(str).tolist()

        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                input_tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top5 = torch.topk(probs, 5)
                predictions.append({
                    'image': path,
                    'predicted_class': top5.indices[0].item(),
                    'confidence': top5.values[0].item(),
                    'top5_classes': top5.indices.tolist(),
                    'top5_probs': [round(p, 4) for p in top5.values.tolist()],
                })
            except Exception as e:
                predictions.append({'image': path, 'error': str(e)})

        out = pd.DataFrame(predictions)
        results['output_text'].append("PROC CVISION - Image Classification")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Model: {model_name}")
        results['output_text'].append(f"Processed {len(image_paths)} images")
        results['output_data'] = out
        return results

    # ==================================================================
    # MODE: generate_samples
    # ==================================================================
    def _generate_samples(
        self, proc_info: ProcStatement, **kw
    ) -> Dict[str, Any]:
        """Create synthetic images with coloured shapes + bounding-box annotations."""
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}

        n_train = int(proc_info.options.get('n_train', 30))
        n_test = int(proc_info.options.get('n_test', 10))
        outdir = str(
            proc_info.options.get('outdir', './examples/work/objdet_images')
        ).strip("'\"")
        seed = int(proc_info.options.get('seed', 42))
        img_size = int(proc_info.options.get('img_size', 128))

        random.seed(seed)
        np.random.seed(seed)

        os.makedirs(os.path.join(outdir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(outdir, 'test'), exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for split, n in [('train', n_train), ('test', n_test)]:
            for i in range(n):
                img, objects = self._make_one_image(
                    img_size, seed + i + (0 if split == 'train' else 100000)
                )
                fname = f'{split}_{i:04d}.png'
                fpath = os.path.join(outdir, split, fname)
                img.save(fpath)
                for obj in objects:
                    rows.append({
                        'image_path': os.path.abspath(fpath),
                        'x_min': obj['x_min'],
                        'y_min': obj['y_min'],
                        'x_max': obj['x_max'],
                        'y_max': obj['y_max'],
                        'label': obj['label'],
                        'split': split,
                    })

        out_df = pd.DataFrame(rows)

        results['output_text'].append(
            "PROC CVISION - Generate Object Detection Samples"
        )
        results['output_text'].append("=" * 50)
        results['output_text'].append(
            f"Generated {n_train} training + {n_test} test images"
        )
        results['output_text'].append(f"Image size: {img_size}x{img_size}")
        results['output_text'].append(f"Output directory: {outdir}")
        results['output_text'].append(f"Total annotations: {len(out_df)}")
        results['output_text'].append(
            f"Classes: {sorted(out_df['label'].unique().tolist())}"
        )
        results['output_data'] = out_df
        results['output_dataset'] = 'annotations'
        return results

    @staticmethod
    def _make_one_image(
        img_size: int, seed: int
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """Draw 1-3 random shapes on a random background."""
        rng = random.Random(seed)

        bg_choices = [
            (255, 255, 255),
            (240, 240, 240),
            (245, 245, 220),
            (200, 220, 240),
            (230, 230, 250),
        ]
        bg = bg_choices[rng.randint(0, len(bg_choices) - 1)]
        img = Image.new('RGB', (img_size, img_size), bg)
        draw = ImageDraw.Draw(img)

        shapes = list(_SHAPE_PALETTE.keys())
        n_objects = rng.randint(1, 3)
        objects: List[Dict[str, Any]] = []

        min_size = max(10, img_size // 8)
        max_size = max(min_size + 1, min(50, img_size // 2))

        for _ in range(n_objects):
            shape = rng.choice(shapes)
            size = rng.randint(min_size, max_size)
            x1 = rng.randint(2, max(3, img_size - size - 2))
            y1 = rng.randint(2, max(3, img_size - size - 2))
            x2, y2 = x1 + size, y1 + size

            fill, outline = _SHAPE_PALETTE[shape]

            if shape == 'circle':
                draw.ellipse(
                    [x1, y1, x2, y2], fill=fill, outline=outline, width=2
                )
            elif shape == 'rectangle':
                draw.rectangle(
                    [x1, y1, x2, y2], fill=fill, outline=outline, width=2
                )
            elif shape == 'triangle':
                cx = (x1 + x2) / 2
                draw.polygon(
                    [(cx, y1), (x1, y2), (x2, y2)],
                    fill=fill,
                    outline=outline,
                )

            objects.append({
                'x_min': x1,
                'y_min': y1,
                'x_max': x2,
                'y_max': y2,
                'label': shape,
            })

        return img, objects

    # ==================================================================
    # MODE: train_detect
    # ==================================================================
    def _train_detect(
        self, data: pd.DataFrame, proc_info: ProcStatement, **kw
    ) -> Dict[str, Any]:
        """Fine-tune Faster R-CNN on bounding-box annotations."""
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}

        if not HAS_DETECTION:
            results['output_text'].append(
                "ERROR: torchvision detection models not available. "
                "Install torchvision>=0.13."
            )
            return results

        # ---- options -------------------------------------------------
        image_col = str(proc_info.options.get('image', 'image_path'))
        epochs = int(proc_info.options.get('epochs', 5))
        lr = float(proc_info.options.get('lr', 0.005))
        batch_size = int(proc_info.options.get('batch_size', 4))
        model_id = str(proc_info.options.get('model_name', 'objdet_model'))
        backbone = str(
            proc_info.options.get('backbone', 'resnet50')
        ).lower()

        # ---- required columns ----------------------------------------
        bbox_cols = ['x_min', 'y_min', 'x_max', 'y_max']
        label_col = 'label'
        required = [image_col] + bbox_cols + [label_col]
        missing = [c for c in required if c not in data.columns]
        if missing:
            results['output_text'].append(
                f"ERROR: Missing columns: {missing}"
            )
            return results

        # ---- label mapping (0 = background) --------------------------
        unique_labels = sorted(data[label_col].unique().tolist())
        label_to_id = {lbl: idx + 1 for idx, lbl in enumerate(unique_labels)}
        id_to_label = {v: k for k, v in label_to_id.items()}
        num_classes = len(unique_labels) + 1

        # ---- group annotations by image ------------------------------
        grouped = data.groupby(image_col)
        image_paths = list(grouped.groups.keys())

        if len(image_paths) < 2:
            results['output_text'].append(
                "ERROR: At least 2 training images required"
            )
            return results

        # ---- build model ---------------------------------------------
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        device = torch.device('cpu')
        model.to(device)
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        )
        to_tensor = transforms.ToTensor()

        # ---- training loop -------------------------------------------
        epoch_losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            indices = list(range(len(image_paths)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_idx = indices[batch_start : batch_start + batch_size]
                images: List[Any] = []
                targets: List[Dict[str, Any]] = []

                for idx in batch_idx:
                    img_path = image_paths[idx]
                    grp = grouped.get_group(img_path)
                    try:
                        img = Image.open(str(img_path)).convert('RGB')
                        img_t = to_tensor(img).to(device)
                    except Exception:
                        continue

                    boxes = torch.FloatTensor(
                        grp[bbox_cols].values.astype(float)
                    ).to(device)
                    labels = torch.LongTensor(
                        [label_to_id[lbl] for lbl in grp[label_col].values]
                    ).to(device)

                    images.append(img_t)
                    targets.append({'boxes': boxes, 'labels': labels})

                if not images:
                    continue

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

        # ---- save to model store -------------------------------------
        model_obj = {
            'state_dict': model.state_dict(),
            'num_classes': num_classes,
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'backbone': backbone,
        }

        results['model_object'] = model_obj
        results['model_name'] = model_id
        results['model_metadata'] = {
            'proc': 'CVISION',
            'mode': 'train_detect',
            'num_classes': num_classes,
            'labels': unique_labels,
            'epochs': epochs,
            'final_loss': epoch_losses[-1] if epoch_losses else 0,
        }

        # ---- metrics output ------------------------------------------
        metrics_df = pd.DataFrame({
            'epoch': list(range(1, epochs + 1)),
            'loss': epoch_losses,
        })

        results['output_text'].append(
            "PROC CVISION - Object Detection Training (Faster R-CNN)"
        )
        results['output_text'].append("=" * 55)
        results['output_text'].append(f"Backbone: Faster R-CNN ({backbone})")
        results['output_text'].append(
            f"Classes: {unique_labels} ({num_classes} incl. background)"
        )
        results['output_text'].append(
            f"Training images: {len(image_paths)}"
        )
        results['output_text'].append(f"Epochs: {epochs}  |  LR: {lr}")
        results['output_text'].append("")
        results['output_text'].append("Training Loss")
        results['output_text'].append("-" * 30)
        for i, loss in enumerate(epoch_losses):
            results['output_text'].append(
                f"  Epoch {i + 1:>3}/{epochs}  loss = {loss:.4f}"
            )
        results['output_text'].append("")
        results['output_text'].append(f"Model saved as: '{model_id}'")
        results['output_data'] = metrics_df
        return results

    # ==================================================================
    # MODE: detect / score / serve
    # ==================================================================
    def _detect(
        self, data: pd.DataFrame, proc_info: ProcStatement, **kw
    ) -> Dict[str, Any]:
        """Run object detection on images — pretrained or from model store."""
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}

        if not HAS_DETECTION:
            results['output_text'].append(
                "ERROR: torchvision detection models not available"
            )
            return results

        # ---- options -------------------------------------------------
        image_col = str(proc_info.options.get('image', 'image_path'))
        confidence = float(proc_info.options.get('confidence', 0.5))
        model_id = str(proc_info.options.get('model_name', ''))

        if image_col not in data.columns:
            str_cols = data.select_dtypes(include='object').columns.tolist()
            if str_cols:
                image_col = str_cols[0]
            else:
                results['output_text'].append("ERROR: IMAGE column required")
                return results

        # ---- load model ----------------------------------------------
        model_store = kw.get('model_store')
        id_to_label: Dict[int, str] | None = None
        num_classes = 91  # COCO default
        source_label = "pretrained COCO"

        if model_id and model_store:
            stored = model_store.load(model_id)
            if stored and stored.model:
                model_data = stored.model
                state_dict = model_data.get('state_dict')
                num_classes = model_data.get('num_classes', 91)
                id_to_label = model_data.get('id_to_label')

                model = fasterrcnn_resnet50_fpn(weights=None)
                in_features = (
                    model.roi_heads.box_predictor.cls_score.in_features
                )
                model.roi_heads.box_predictor = FastRCNNPredictor(
                    in_features, num_classes
                )
                model.load_state_dict(state_dict)
                source_label = f"model '{model_id}'"
            else:
                results['output_text'].append(
                    f"WARNING: Model '{model_id}' not found in store — "
                    "falling back to pretrained COCO"
                )
                model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        else:
            model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

        model.eval()
        device = torch.device('cpu')
        model.to(device)
        to_tensor = transforms.ToTensor()

        # ---- run detection on each unique image ----------------------
        image_paths = data[image_col].dropna().unique().tolist()
        detection_rows: List[Dict[str, Any]] = []

        for path in image_paths:
            try:
                img = Image.open(str(path)).convert('RGB')
                img_tensor = to_tensor(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    preds = model(img_tensor)

                pred = preds[0]
                boxes = pred['boxes'].cpu()
                labels = pred['labels'].cpu()
                scores = pred['scores'].cpu()

                mask = scores >= confidence
                boxes = boxes[mask]
                labels = labels[mask]
                scores = scores[mask]

                for j in range(len(boxes)):
                    box = boxes[j].tolist()
                    label_id = labels[j].item()
                    score = scores[j].item()
                    label_str = (
                        id_to_label.get(label_id, str(label_id))
                        if id_to_label
                        else str(label_id)
                    )
                    detection_rows.append({
                        'image_path': str(path),
                        'x_min': round(box[0], 1),
                        'y_min': round(box[1], 1),
                        'x_max': round(box[2], 1),
                        'y_max': round(box[3], 1),
                        'predicted_label': label_str,
                        'confidence': round(score, 4),
                    })

                if not mask.any():
                    detection_rows.append({
                        'image_path': str(path),
                        'x_min': None,
                        'y_min': None,
                        'x_max': None,
                        'y_max': None,
                        'predicted_label': 'none',
                        'confidence': 0.0,
                    })

            except Exception:
                detection_rows.append({
                    'image_path': str(path),
                    'x_min': None,
                    'y_min': None,
                    'x_max': None,
                    'y_max': None,
                    'predicted_label': 'error',
                    'confidence': 0.0,
                })

        out_df = pd.DataFrame(detection_rows)

        n_detected = len(
            out_df[out_df['predicted_label'].isin(['none', 'error']) == False]  # noqa: E712
        )
        results['output_text'].append("PROC CVISION - Object Detection")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Model source: {source_label}")
        results['output_text'].append(
            f"Confidence threshold: {confidence}"
        )
        results['output_text'].append(
            f"Images processed: {len(image_paths)}"
        )
        results['output_text'].append(f"Detections found: {n_detected}")
        if id_to_label:
            labels_found = sorted(
                out_df.loc[
                    ~out_df['predicted_label'].isin(['none', 'error']),
                    'predicted_label',
                ]
                .unique()
                .tolist()
            )
            results['output_text'].append(
                f"Labels detected: {labels_found}"
            )
        results['output_data'] = out_df
        return results
