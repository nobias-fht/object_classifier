"""
object_classifier.py
--------------------
Train and apply a random-forest classifier on objects in a 3D label image.

Workflow
--------
1. Extract per-object features from scikit-image regionprops (all scalar 3D
   properties are attempted; any that fail are silently dropped).
2. Build a training set from a user-supplied class dictionary.
3. Fit a scikit-learn RandomForestClassifier.
4. Predict classes for every object and return a colour-coded label image.

Example
-------
    from object_classifier import ObjectClassifier

    classes = {
        'neuron': [1, 3, 7],
        'glia':   [2, 5],
    }

    clf = ObjectClassifier()
    clf.fit(raw, labels, classes)

    result, class_names = clf.predict(raw, labels)
    # result[i, j, k] == 1  →  class_names[0]
    # result[i, j, k] == 2  →  class_names[1]
    # result[i, j, k] == 0  →  background
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# All features `extract_features` can produce (3D-compatible; properties
# that require 2D, e.g. perimeter/eccentricity, are excluded), with
# human-readable descriptions for the GUI feature-selection widget.
#
# Features flagged "expensive" involve a convex-hull computation (or a
# similarly costly operation) and can dominate runtime when there are
# many/large objects -- especially in `fit_from_annotations`, where every
# saved training image is re-processed on every retrain.
# ---------------------------------------------------------------------------
FEATURE_INFO: dict[str, dict] = {
    "area": {
        "description": "Number of voxels in the object.",
        "expensive": False,
    },
    "area_bbox": {
        "description": "Number of voxels in the object's bounding box.",
        "expensive": False,
    },
    "area_convex": {
        "description": "Number of voxels in the convex hull of the object (expensive).",
        "expensive": True,
    },
    "area_filled": {
        "description": "Number of voxels in the object with internal holes filled.",
        "expensive": False,
    },
    "axis_major_length": {
        "description": "Length of the major axis of the ellipsoid with the same "
                        "normalized second central moments as the object.",
        "expensive": False,
    },
    "axis_minor_length": {
        "description": "Length of the minor axis of the ellipsoid with the same "
                        "normalized second central moments as the object.",
        "expensive": False,
    },
    "equivalent_diameter_area": {
        "description": "Diameter of a sphere with the same volume as the object.",
        "expensive": False,
    },
    "euler_number": {
        "description": "Euler characteristic (connectivity / hole count) of the object (expensive).",
        "expensive": True,
    },
    "extent": {
        "description": "Ratio of object voxels to bounding-box voxels.",
        "expensive": False,
    },
    "feret_diameter_max": {
        "description": "Maximum caliper diameter, computed from the convex hull (expensive).",
        "expensive": True,
    },
    "solidity": {
        "description": "Ratio of object volume to convex-hull volume (expensive).",
        "expensive": True,
    },
    "intensity_max": {
        "description": "Maximum raw intensity within the object.",
        "expensive": False,
    },
    "intensity_mean": {
        "description": "Mean raw intensity within the object.",
        "expensive": False,
    },
    "intensity_min": {
        "description": "Minimum raw intensity within the object.",
        "expensive": False,
    },
    "intensity_std": {
        "description": "Standard deviation of raw intensity within the object.",
        "expensive": False,
    },
    "inertia_ev_0": {
        "description": "Largest eigenvalue of the inertia tensor.",
        "expensive": False,
    },
    "inertia_ev_1": {
        "description": "Middle eigenvalue of the inertia tensor.",
        "expensive": False,
    },
    "inertia_ev_2": {
        "description": "Smallest eigenvalue of the inertia tensor.",
        "expensive": False,
    },
    "axis_ratio": {
        "description": "Ratio of minor to major axis length.",
        "expensive": False,
    },
}

# Underlying regionprops propert(y/ies) each derived feature depends on.
# Feature names not listed here map 1:1 to a regionprops property of the
# same name.
_FEATURE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "inertia_ev_0": ("inertia_tensor_eigvals",),
    "inertia_ev_1": ("inertia_tensor_eigvals",),
    "inertia_ev_2": ("inertia_tensor_eigvals",),
    "axis_ratio": ("axis_major_length", "axis_minor_length"),
}


def extract_features(
    labels: np.ndarray, raw: np.ndarray, feature_names: list[str] | None = None
) -> pd.DataFrame:
    """
    Extract per-object features from a label image using regionprops.

    Each regionprops property is attempted independently; properties that raise
    an exception for any object are dropped from the output entirely.
    The inertia-tensor eigenvalues (3 values per object in 3D) are expanded
    into separate columns ``inertia_ev_0``, ``inertia_ev_1``, ``inertia_ev_2``.
    An ``axis_ratio`` column (minor / major axis length) is also added.

    Parameters
    ----------
    labels : ndarray, shape (Z, Y, X)
        Integer label image; 0 is background.
    raw : ndarray, shape (Z, Y, X)
        Intensity image used for intensity-based features.
    feature_names : list of str, optional
        Subset of ``FEATURE_INFO`` keys to compute. Only the regionprops
        properties needed for these features are computed, so restricting
        this list (e.g. excluding features flagged "expensive" in
        ``FEATURE_INFO``) speeds up extraction. If ``None`` (default), every
        feature in ``FEATURE_INFO`` is computed.

    Returns
    -------
    df : pd.DataFrame
        One row per object.  The ``label`` column holds the original label ID.
    """
    if feature_names is None:
        feature_names = list(FEATURE_INFO.keys())
    else:
        unknown = set(feature_names) - set(FEATURE_INFO)
        if unknown:
            raise ValueError(f"Unknown feature name(s): {sorted(unknown)}")

    needs_inertia = any(name.startswith("inertia_ev_") for name in feature_names)
    needs_axis_ratio = "axis_ratio" in feature_names

    direct_props = {name for name in feature_names if name not in _FEATURE_DEPENDENCIES}
    if needs_axis_ratio:
        direct_props.update(_FEATURE_DEPENDENCIES["axis_ratio"])
    # Preserve a stable, deterministic column order.
    scalar_props = tuple(name for name in FEATURE_INFO if name in direct_props)

    all_props = ("label",) + scalar_props
    if needs_inertia:
        all_props = all_props + ("inertia_tensor_eigvals",)

    data: dict[str, np.ndarray] = {}

    # Fast path: try to compute every requested property in a single
    # regionprops_table call (one pass over the image).  Fall back to
    # property-by-property extraction only if any property causes the batch
    # call to fail.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = regionprops_table(labels, intensity_image=raw, properties=all_props)
        data["label"] = table["label"]
        n_objects = len(data["label"])
        for prop in scalar_props:
            if prop in table:
                values = table[prop]
                if len(values) == n_objects and np.issubdtype(values.dtype, np.number):
                    data[prop] = values
        if needs_inertia:
            for i in range(3):
                col = f"inertia_tensor_eigvals-{i}"
                if col in table and len(table[col]) == n_objects:
                    data[f"inertia_ev_{i}"] = table[col]

    except Exception:
        # Slow fallback: attempt each property independently so that a single
        # failing property does not block the others.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = regionprops_table(labels, intensity_image=raw, properties=("label",))
        data["label"] = base["label"]
        n_objects = len(data["label"])

        for prop in scalar_props:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    table = regionprops_table(
                        labels, intensity_image=raw, properties=("label", prop)
                    )
                values = table[prop]
                if len(values) == n_objects and np.issubdtype(values.dtype, np.number):
                    data[prop] = values
            except Exception:
                pass

        if needs_inertia:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    table = regionprops_table(
                        labels, intensity_image=raw,
                        properties=("label", "inertia_tensor_eigvals"),
                    )
                for i in range(3):
                    col = f"inertia_tensor_eigvals-{i}"
                    if col in table and len(table[col]) == n_objects:
                        data[f"inertia_ev_{i}"] = table[col]
            except Exception:
                pass

    df = pd.DataFrame(data)

    # Derived feature: axis ratio (minor / major length)
    if needs_axis_ratio and "axis_major_length" in df.columns and "axis_minor_length" in df.columns:
        major = df["axis_major_length"].replace(0, np.nan)
        df["axis_ratio"] = df["axis_minor_length"] / major

    # Keep only the requested output columns, dropping any columns that were
    # computed solely as a dependency of a derived feature (e.g.
    # axis_major_length/axis_minor_length pulled in only for axis_ratio).
    output_cols = [name for name in feature_names if name in df.columns]
    df = df[["label"] + output_cols]

    # Replace inf/-inf and NaN with the column mean.
    # Infinities arise from divisions (e.g. axis_ratio when major_length == 0)
    # or from degenerate regionprops computations.
    feature_cols = [c for c in df.columns if c != "label"]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    return df


def _lookup_class_indices(
    mask: np.ndarray, train_labels: np.ndarray, object_ids: np.ndarray
) -> np.ndarray:
    """
    Look up the annotated class index for each object.

    This assumes every voxel belonging to a given object was painted with the
    same class index (which is how the training GUI assigns classes: clicking
    an object fills its entire footprint at once), so it simply reads the
    ``train_labels`` value at one annotated voxel per object.

    Parameters
    ----------
    mask : ndarray
        Integer object-label image (0 = background).
    train_labels : ndarray, same shape as ``mask``
        Integer image where annotated objects are filled with their class
        index (1..N), and unannotated voxels are 0.
    object_ids : array of int
        Object IDs (from ``mask``) to look up, in the order the result should
        be returned.

    Returns
    -------
    class_indices : ndarray of int
        Class index for each object ID (0 if the object is not annotated).
    """
    mask_flat = mask.ravel()
    train_labels_flat = train_labels.ravel()

    annotated = train_labels_flat > 0
    id_to_class_index = dict(zip(mask_flat[annotated], train_labels_flat[annotated]))

    return np.array(
        [id_to_class_index.get(object_id, 0) for object_id in object_ids], dtype=int
    )


class ObjectClassifier:
    """
    Random-forest classifier for objects in a 3D label image.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the random forest.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.forest = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.class_names: list[str] | None = None
        self.feature_columns: list[str] | None = None
        self.feature_means_: dict[str, float] = {}

    def fit(
        self,
        raw: np.ndarray,
        labels: np.ndarray,
        classes: dict[str, list[int]],
        features: list[str] | None = None,
        crop_to_training: bool = True,
    ) -> "ObjectClassifier":
        """
        Extract features and train the classifier on labelled examples.

        Parameters
        ----------
        raw : ndarray
            Intensity image.
        labels : ndarray
            Integer label image (0 = background).
        classes : dict
            ``{'class_name': [label_id, ...], ...}``
            Only objects whose label ID appears in this dict are used for
            training; all others are classified at prediction time.
        features : list of str, optional
            Subset of ``FEATURE_INFO`` feature names to compute and train on.
            If ``None`` (default), every feature in ``FEATURE_INFO`` is used.
            Pass the ``'feature'`` column from ``feature_importances()``
            (e.g. the top-10 entries) to restrict training to those features
            and speed up both fitting and inference.
        crop_to_training : bool, optional
            If ``True`` (default), feature extraction is restricted to the
            tight bounding box containing only the training objects, which
            is much faster on large images.  Set to ``False`` to run
            regionprops over the full image instead.

        Returns
        -------
        self
        """
        # Map every label ID to its class name
        label_to_class: dict[int, str] = {}
        for class_name, label_ids in classes.items():
            for lid in label_ids:
                label_to_class[lid] = class_name

        if crop_to_training:
            # Speed up feature extraction by restricting to the bounding box
            # containing only the training objects.  This avoids running
            # regionprops over the full image (which may contain thousands of
            # irrelevant objects).
            all_training_ids = list(label_to_class.keys())
            training_voxel_mask = np.isin(labels, all_training_ids)
            if training_voxel_mask.any():
                coords = np.argwhere(training_voxel_mask)
                bbox = tuple(
                    slice(lo, hi)
                    for lo, hi in zip(coords.min(axis=0), coords.max(axis=0) + 1)
                )
                labels_fit = np.where(training_voxel_mask, labels, 0)[bbox]
                raw_fit = raw[bbox]
            else:
                labels_fit = np.where(training_voxel_mask, labels, 0)
                raw_fit = raw
        else:
            labels_fit = labels
            raw_fit = raw

        features_df = extract_features(labels_fit, raw_fit, feature_names=features)

        # Filter to training objects only
        train_mask = features_df["label"].isin(label_to_class)
        train_df = features_df[train_mask].copy()
        train_df["class"] = train_df["label"].map(label_to_class)

        if train_df.empty:
            raise ValueError(
                "None of the label IDs in `classes` were found in the label image."
            )

        # Feature matrix and target vector
        self.class_names = sorted(classes.keys())
        self.feature_columns = [c for c in train_df.columns if c not in ("label", "class")]
        if not self.feature_columns:
            raise ValueError("No features were selected/extracted for training.")

        X_df = train_df[self.feature_columns]
        self.feature_means_ = X_df.mean().to_dict()
        X = X_df.values
        y = train_df["class"].values

        print(f"Training on {len(train_df)} objects, {len(self.feature_columns)} features, "
              f"{len(self.class_names)} classes: {self.class_names}")
        self.forest.fit(X, y)
        print("Done.")
        return self

    def fit_from_annotations(
        self,
        triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        class_names: list[str],
        features: list[str] | None = None,
        crop_to_training: bool = True,
    ) -> "ObjectClassifier":
        """
        Train the classifier from one or more painted (raw, mask, train_labels)
        triplets, such as the ones produced by the napari training GUI.

        Unlike ``fit()``, which takes a single image and a ``{class_name: [label_ids]}``
        dictionary, this method combines training examples from multiple images at
        once, where each object's class is read directly from a painted label image.

        Parameters
        ----------
        triplets : list of (raw, mask, train_labels) tuples
            raw : intensity image
            mask : integer object-label image (0 = background)
            train_labels : integer image, same shape as ``mask``, where every voxel
                of an annotated object is filled with its class index (1..N), and
                unannotated voxels are 0.
        class_names : list of str
            Ordered class names. ``class_names[i]`` corresponds to class index
            ``i + 1`` in the ``train_labels`` images.
        features : list of str, optional
            Subset of ``FEATURE_INFO`` feature names to compute and train on.
            If ``None`` (default), every feature in ``FEATURE_INFO`` is used.
        crop_to_training : bool, optional
            If ``True`` (default), feature extraction for each triplet is
            restricted to the tight bounding box containing only the
            annotated objects in that image. Without this, every object in
            every saved image (annotated or not) gets its features computed
            on every retrain, which scales poorly as the number of saved
            training images grows. Set to ``False`` to run regionprops over
            each full image instead.

        Returns
        -------
        self
        """
        self.class_names = list(class_names)

        all_annotated_rows = []
        for raw, mask, train_labels in triplets:
            if crop_to_training:
                # Speed up feature extraction by restricting to the bounding
                # box containing only the annotated objects in this image,
                # mirroring the cropping done in fit().
                annotated_voxel_mask = train_labels > 0
                if annotated_voxel_mask.any():
                    coords = np.argwhere(annotated_voxel_mask)
                    bbox = tuple(
                        slice(lo, hi)
                        for lo, hi in zip(coords.min(axis=0), coords.max(axis=0) + 1)
                    )
                    mask_fit = np.where(annotated_voxel_mask, mask, 0)[bbox]
                    raw_fit = raw[bbox]
                    train_labels_fit = train_labels[bbox]
                else:
                    mask_fit = np.where(annotated_voxel_mask, mask, 0)
                    raw_fit = raw
                    train_labels_fit = train_labels
            else:
                mask_fit = mask
                raw_fit = raw
                train_labels_fit = train_labels

            features_df = extract_features(mask_fit, raw_fit, feature_names=features)
            class_indices = _lookup_class_indices(
                mask_fit, train_labels_fit, features_df["label"].values
            )
            features_df = features_df.copy()
            features_df["class_idx"] = class_indices
            annotated_df = features_df[features_df["class_idx"] > 0]
            all_annotated_rows.append(annotated_df)

        train_df = pd.concat(all_annotated_rows, ignore_index=True)
        if train_df.empty:
            raise ValueError(
                "None of the objects in the given triplets have been annotated."
            )

        self.feature_columns = [c for c in train_df.columns if c not in ("label", "class_idx")]
        if not self.feature_columns:
            raise ValueError("No features were selected/extracted for training.")

        X_df = train_df[self.feature_columns]
        self.feature_means_ = X_df.mean().to_dict()
        X = X_df.values

        class_idx_to_name = {i + 1: name for i, name in enumerate(self.class_names)}
        y = [class_idx_to_name[idx] for idx in train_df["class_idx"].values]

        print(f"Training on {len(train_df)} objects from {len(triplets)} image(s), "
              f"{len(self.feature_columns)} features, "
              f"{len(self.class_names)} classes: {self.class_names}")
        self.forest.fit(X, y)
        print("Done.")
        return self

    def predict(
        self,
        raw: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Classify every object in the label image.

        Parameters
        ----------
        raw : ndarray
            Intensity image.
        labels : ndarray
            Integer label image (0 = background).

        Returns
        -------
        result : ndarray (same shape as labels)
            Each object's voxels are filled with its class index (1-based).
            Background = 0.  Index i corresponds to ``class_names[i - 1]``.
        class_names : list of str
            Ordered class names.  ``class_names[i - 1]`` is the class for
            voxels with value ``i`` in ``result``.
        """
        if self.feature_columns is None:
            raise RuntimeError("Call fit() before predict().")

        features_df = extract_features(labels, raw, feature_names=self.feature_columns)

        # Use only the features seen during training.
        # Fill any features that were present at training time but couldn't be
        # computed on this image (e.g. due to image properties or object count)
        # with the mean value recorded during fit().
        missing = set(self.feature_columns) - set(features_df.columns)
        if missing:
            print(f"Warning: {len(missing)} feature(s) missing at inference time, "
                  f"filling with training means: {missing}")
            for col in missing:
                features_df[col] = self.feature_means_.get(col, 0.0)

        X = features_df[self.feature_columns].values
        predicted_classes = self.forest.predict(X)

        # Map class names → 1-based integer index
        class_to_idx = {name: i + 1 for i, name in enumerate(self.class_names)}

        # Build a lookup table (label_id → class_index) and apply it with a
        # single vectorised index operation instead of one boolean-mask scan
        # per object.  This is O(image_size) rather than O(n_objects × size).
        max_label = int(labels.max())
        lut = np.zeros(max_label + 1, dtype=np.result_type(len(self.class_names), np.uint8))
        class_indices = np.array(
            [class_to_idx[c] for c in predicted_classes], dtype=lut.dtype
        )
        lut[features_df["label"].values] = class_indices
        result = lut[labels]

        return result, self.class_names

    def save(self, path: str | Path) -> None:
        """
        Save this fitted classifier (forest, class names, feature columns,
        and feature means) to disk with joblib.
        """
        joblib.dump(
            {
                "forest": self.forest,
                "class_names": self.class_names,
                "feature_columns": self.feature_columns,
                "feature_means_": self.feature_means_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ObjectClassifier":
        """Load a classifier previously saved with :meth:`save`."""
        state = joblib.load(path)
        clf = cls()
        clf.forest = state["forest"]
        clf.class_names = state["class_names"]
        clf.feature_columns = state["feature_columns"]
        clf.feature_means_ = state["feature_means_"]
        return clf

    def feature_importances(self) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances, sorted descending.

        Returns
        -------
        pd.DataFrame with columns ``feature`` and ``importance``.
        """
        if self.feature_columns is None:
            raise RuntimeError("Call fit() before feature_importances().")
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": self.forest.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
