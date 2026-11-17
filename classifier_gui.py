"""
classifier_gui.py
------------------
A napari-based GUI for interactively training an ObjectClassifier.

Folder layout expected
-----------------------
Two folders are chosen in the GUI: one with raw images and one with the
matching object masks, using identical file names::

    project/
      images/   cell_01.tif  cell_02.tif ...
      labels/   cell_01.tif  cell_02.tif ...
      train_labels/   (created automatically next to ``labels/``)

When an image has been annotated and "finalized", its training labels are
saved to ``train_labels/`` under the same file name.

Class names
-----------
Class names are read from a ``config.yaml`` file (next to this script, or a
path given as the first command-line argument), which must look like::

    class_names:
      - alive
      - dead

Workflow
--------
1. Choose the images and labels folders, then click "Start". The first image
   without saved training labels is loaded, along with its mask, into the
   napari viewer.
   An empty "training" labels layer holds your annotations, and a
   "prediction" layer shows classifier output as outlines only.
2. Click on objects to assign them to the class selected in the "Current
   class" dropdown. The clicked object's whole footprint is filled with that
   class in the "training" layer. Only clicked objects become training data.
3. Once a few objects per class have been annotated, click "Train". This
   fits the classifier on all clicked objects (previously saved images plus
   the current one) and draws its predictions for the current image as
   outlines in the "prediction" layer. The "training" layer is not changed.
4. Click objects whose predicted outline is wrong or that are otherwise
   informative to add them to the training set. Clicking an object already
   in the selected class removes it again.
5. Click "Finalize" to save the "training" layer to disk and move on to the
   next unlabeled image, which is classified (as outlines) using the freshly
   retrained classifier.

Run this file directly to launch napari with the widget docked:

    python classifier_gui.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import skimage.io
import skimage.measure
import yaml
import napari
from magicgui.widgets import Container, PushButton, ComboBox, CheckBox, Label, FileEdit

from object_classifier import FEATURE_INFO, ObjectClassifier

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_class_names(config_path: Path) -> list[str]:
    """
    Read the ``class_names`` list from a config YAML file.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the file does not contain a ``class_names`` list with at least
        two entries.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}

    names = config.get("class_names")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(
            f"{config_path} must contain a 'class_names' list of strings."
        )

    names = [name.strip() for name in names if name.strip()]
    if len(names) < 2:
        raise ValueError(
            f"{config_path} 'class_names' must have at least two entries."
        )
    return names


# ---------------------------------------------------------------------------
# File-naming helpers
# ---------------------------------------------------------------------------

def ensure_instance_labels(mask: np.ndarray) -> np.ndarray:
    """
    Ensure a mask array contains per-object instance labels.

    If the mask only contains values 0 and 1 (i.e. it is a semantic
    segmentation / binary mask rather than an instance segmentation), it is
    relabeled with ``skimage.measure.label`` so each connected component
    gets its own id. Otherwise the mask is returned unchanged.
    """
    if np.amax(mask) <= 1:
        return skimage.measure.label(mask)
    return mask


@dataclass(frozen=True)
class DatasetFolders:
    """
    The folders making up a training dataset.

    ``images`` holds the raw images and ``labels`` holds the object masks
    with identical file names. Annotations are saved to a ``train_labels``
    folder next to ``labels``, again using the same file names.
    """

    images: Path
    labels: Path

    @property
    def train_labels(self) -> Path:
        return self.labels.parent / "train_labels"


def get_mask_path(image_path: Path, folders: DatasetFolders) -> Path:
    """Return the expected mask file path for a given training image."""
    return folders.labels / image_path.name


def get_train_labels_path(image_path: Path, folders: DatasetFolders) -> Path:
    """Return the expected saved-training-labels file path for an image."""
    return folders.train_labels / image_path.name


def list_image_files(folders: DatasetFolders) -> list[Path]:
    """
    Return a sorted list of image files in the images folder that have a
    same-named file in the labels folder.
    """
    candidates = sorted(
        list(folders.images.glob("*.tif")) + list(folders.images.glob("*.tiff"))
    )
    return [path for path in candidates if get_mask_path(path, folders).exists()]


def find_next_image(
    folders: DatasetFolders, after: Path | None = None, ignore_labeled: bool = True
) -> Path | None:
    """
    Return the next image to load from the sorted image list.

    Parameters
    ----------
    folders : DatasetFolders
        Images, labels and training-labels folders.
    after : Path, optional
        If given, only images that come after this one (in sorted folder
        order) are considered, e.g. to advance to the image following the
        one currently open. If None (default), scan from the start of the
        folder.
    ignore_labeled : bool, optional
        If True (default), skip any image that already has a saved
        training-labels file. If False, return images in folder order
        regardless of whether they already have saved labels.

    Returns
    -------
    Path or None
        The next matching image, or None if there isn't one.
    """
    images = list_image_files(folders)
    if after is not None:
        try:
            images = images[images.index(after) + 1 :]
        except ValueError:
            pass

    for image_path in images:
        if ignore_labeled and get_train_labels_path(image_path, folders).exists():
            continue
        return image_path
    return None


def find_saved_triplets(
    folders: DatasetFolders, skip: Path | None = None
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load every (raw, mask, train_labels) triplet already saved to disk.

    Parameters
    ----------
    folders : DatasetFolders
        Images, labels and training-labels folders.
    skip : Path, optional
        An image path to exclude, e.g. the currently open image, whose
        in-progress annotations should be added separately.
    """
    triplets = []
    for image_path in list_image_files(folders):
        if skip is not None and image_path == skip:
            continue
        train_labels_path = get_train_labels_path(image_path, folders)
        if not train_labels_path.exists():
            continue
        raw = skimage.io.imread(str(image_path))
        mask = ensure_instance_labels(
            skimage.io.imread(str(get_mask_path(image_path, folders)))
        )
        train_labels = skimage.io.imread(str(train_labels_path))
        triplets.append((raw, mask, train_labels))
    return triplets


# ---------------------------------------------------------------------------
# The napari dock widget
# ---------------------------------------------------------------------------

class ClassifierGuiWidget(Container):
    """napari dock widget for interactively training an ObjectClassifier."""

    def __init__(self, viewer: napari.Viewer, config_path: Path = DEFAULT_CONFIG_PATH):
        super().__init__()
        self.viewer = viewer

        # Session state
        self.folders: DatasetFolders | None = None
        self.class_names: list[str] = load_class_names(config_path)
        self.current_image_path: Path | None = None
        self.raw_layer = None
        self.mask_layer = None
        self.prediction_layer = None
        self.training_layer = None
        self.classifier: ObjectClassifier | None = None

        # Widgets
        self.images_folder_edit = FileEdit(mode="d", label="Images folder")
        self.labels_folder_edit = FileEdit(mode="d", label="Labels folder")
        self.ignore_labeled_checkbox = CheckBox(
            text="ignore images with training data", value=True
        )
        self.start_button = PushButton(text="Start")
        self.load_image_edit = FileEdit(
            mode="r", label="Image to load", filter="*.tif *.tiff"
        )
        self.load_image_button = PushButton(text="Load image")
        self.current_class_combo = ComboBox(
            label="Current class", choices=self.class_names
        )
        self.train_button = PushButton(text="Train", enabled=False)
        self.finalize_button = PushButton(text="Finalize", enabled=False)
        self.skip_button = PushButton(text="Skip file", enabled=False)
        self.save_classifier_edit = FileEdit(
            mode="w", label="Save classifier to", filter="*.joblib"
        )
        self.save_classifier_button = PushButton(
            text="Save classifier", enabled=False
        )
        self.load_classifier_edit = FileEdit(
            mode="r", label="Load classifier from", filter="*.joblib"
        )
        self.load_classifier_button = PushButton(text="Load classifier")
        self.apply_classifier_button = PushButton(text="Apply classifier", enabled=False)
        self.status_label = Label(
            value=f"Classes: {', '.join(self.class_names)}. Choose the images and labels folders, then click Start."
        )

        # Feature-selection checkboxes, one per entry in FEATURE_INFO.
        # Hidden by default behind `show_features_checkbox`, since most users
        # can rely on the defaults (all features checked).
        self._feature_checkboxes: dict[str, CheckBox] = {}
        for name, info in FEATURE_INFO.items():
            label_text = f"{name} (expensive)" if info["expensive"] else name
            self._feature_checkboxes[name] = CheckBox(
                text=label_text, value=True, tooltip=info["description"]
            )
        self.show_features_checkbox = CheckBox(
            text="Select training features...", value=False
        )
        self.features_container = Container(
            layout="vertical", labels=False, widgets=list(self._feature_checkboxes.values())
        )
        self.features_container.visible = False

        self.start_button.clicked.connect(self._on_start_clicked)
        self.load_image_button.clicked.connect(self._on_load_image_clicked)
        self.train_button.clicked.connect(self._on_train_clicked)
        self.finalize_button.clicked.connect(self._on_finalize_clicked)
        self.skip_button.clicked.connect(self._on_skip_clicked)
        self.save_classifier_button.clicked.connect(self._on_save_classifier_clicked)
        self.load_classifier_button.clicked.connect(self._on_load_classifier_clicked)
        self.apply_classifier_button.clicked.connect(self._on_apply_classifier_clicked)
        self.show_features_checkbox.changed.connect(self._on_show_features_toggled)

        self.viewer.mouse_drag_callbacks.append(self._assign_class_at_click)

        self.extend(
            [
                self.images_folder_edit,
                self.labels_folder_edit,
                self.ignore_labeled_checkbox,
                self.start_button,
                self.load_image_edit,
                self.load_image_button,
                self.current_class_combo,
                self.show_features_checkbox,
                self.features_container,
                self.train_button,
                self.finalize_button,
                self.skip_button,
                self.save_classifier_edit,
                self.save_classifier_button,
                self.load_classifier_edit,
                self.load_classifier_button,
                self.apply_classifier_button,
                self.status_label,
            ]
        )

    # -- Button handlers ---------------------------------------------------

    def _read_folders(self, images_folder: Path | None = None) -> DatasetFolders | None:
        """
        Build a DatasetFolders from the folder inputs, or return None (and set
        a status message) if either folder is invalid.

        ``images_folder`` overrides the images-folder input, e.g. when loading
        a single image.
        """
        if images_folder is None:
            images_folder = Path(str(self.images_folder_edit.value))
        labels_folder = Path(str(self.labels_folder_edit.value))
        if not images_folder.is_dir():
            self.status_label.value = "Please choose a valid images folder."
            return None
        if not labels_folder.is_dir():
            self.status_label.value = "Please choose a valid labels folder."
            return None
        if images_folder.resolve() == labels_folder.resolve():
            self.status_label.value = "Images and labels must be in different folders."
            return None
        return DatasetFolders(images=images_folder, labels=labels_folder)

    def _on_start_clicked(self):
        """Validate the folders, then load the first matching image."""
        folders = self._read_folders()
        if folders is None:
            return

        ignore_labeled = self.ignore_labeled_checkbox.value
        image_path = find_next_image(folders, ignore_labeled=ignore_labeled)
        if image_path is None:
            self.status_label.value = (
                "No unlabeled images with matching labels found."
                if ignore_labeled
                else "No images with matching labels found (file names must match)."
            )
            return

        self.folders = folders

        self._load_image(image_path)
        self.train_button.enabled = True
        self.finalize_button.enabled = True
        self.skip_button.enabled = True

    def _on_load_image_clicked(self):
        """Load a single, specifically chosen image, whether or not it has saved training data."""
        image_path = Path(str(self.load_image_edit.value))
        if not image_path.is_file():
            self.status_label.value = "Please choose a valid image file to load."
            return
        folders = self._read_folders(images_folder=image_path.parent)
        if folders is None:
            return
        if not get_mask_path(image_path, folders).exists():
            self.status_label.value = f"No file named {image_path.name} in the labels folder."
            return

        self.folders = folders

        self._load_image(image_path)
        self.train_button.enabled = True
        self.finalize_button.enabled = True
        self.skip_button.enabled = True
        self.status_label.value = f"Loaded {image_path.name}."

    def _on_train_clicked(self):
        """
        Retrain the classifier on all available training data (saved images
        plus the current in-progress annotations) and re-apply it to the
        current image.
        """
        triplets = self._gather_training_triplets()
        if triplets is None:
            return

        features = self._get_selected_features()
        if features is None:
            return

        self.classifier = ObjectClassifier()
        try:
            self.classifier.fit_from_annotations(triplets, self.class_names, features=features)
        except ValueError as error:
            self.status_label.value = str(error)
            return

        result, _ = self.classifier.predict(self.raw_layer.data, self.mask_layer.data)
        self.prediction_layer.data = result
        self.status_label.value = "Trained and applied classifier to the current image."
        self.save_classifier_button.enabled = True
        self.apply_classifier_button.enabled = True

    def _on_finalize_clicked(self):
        """
        Save the current training labels, retrain the classifier on all
        available data, and load + classify the next unlabeled image.
        """
        train_labels_path = get_train_labels_path(self.current_image_path, self.folders)
        train_labels_path.parent.mkdir(parents=True, exist_ok=True)
        skimage.io.imsave(str(train_labels_path), self.training_layer.data.astype(np.uint8))

        triplets = find_saved_triplets(self.folders)
        if not triplets:
            self.status_label.value = "No training data available to train on."
            return

        features = self._get_selected_features()
        if features is None:
            return

        self.classifier = ObjectClassifier()
        try:
            self.classifier.fit_from_annotations(triplets, self.class_names, features=features)
        except ValueError as error:
            self.status_label.value = str(error)
            return

        next_image_path = find_next_image(
            self.folders,
            after=self.current_image_path,
            ignore_labeled=self.ignore_labeled_checkbox.value,
        )
        if next_image_path is None:
            self.status_label.value = "No more unlabeled images."
            return

        self._load_image(next_image_path)
        result, _ = self.classifier.predict(self.raw_layer.data, self.mask_layer.data)
        self.prediction_layer.data = result
        self.save_classifier_button.enabled = True
        self.apply_classifier_button.enabled = True

    def _on_skip_clicked(self):
        """Load the next image without saving labels, training or predicting."""
        next_image_path = find_next_image(
            self.folders,
            after=self.current_image_path,
            ignore_labeled=self.ignore_labeled_checkbox.value,
        )
        if next_image_path is None:
            self.status_label.value = "No more images to skip to."
            return

        self._load_image(next_image_path)
        self.status_label.value = f"Skipped to {next_image_path.name}."

    def _on_save_classifier_clicked(self):
        """Save the currently trained classifier to the chosen file path."""
        if self.classifier is None:
            self.status_label.value = "Train the classifier before saving it."
            return

        save_path = Path(str(self.save_classifier_edit.value))
        if save_path == Path(".") or save_path.is_dir():
            self.status_label.value = "Please choose a valid file path to save the classifier to."
            return

        self.classifier.save(save_path)
        self.status_label.value = f"Saved classifier to {save_path.name}."

    def _on_load_classifier_clicked(self):
        """Load a previously saved, pre-trained classifier from disk."""
        load_path = Path(str(self.load_classifier_edit.value))
        if not load_path.is_file():
            self.status_label.value = "Please choose a valid classifier file to load."
            return

        try:
            self.classifier = ObjectClassifier.load(load_path)
        except Exception as error:
            self.status_label.value = f"Failed to load classifier: {error}"
            return

        self.apply_classifier_button.enabled = True
        self.status_label.value = f"Loaded classifier from {load_path.name}."

    def _on_apply_classifier_clicked(self):
        """Apply the currently loaded/trained classifier to the open image."""
        if self.classifier is None:
            self.status_label.value = "Load or train a classifier first."
            return
        if self.current_image_path is None:
            self.status_label.value = "Open an image before applying the classifier."
            return

        try:
            result, _ = self.classifier.predict(self.raw_layer.data, self.mask_layer.data)
        except Exception as error:
            self.status_label.value = f"Failed to apply classifier: {error}"
            return

        self.prediction_layer.data = result
        self.status_label.value = "Applied classifier to the current image."

    def _on_show_features_toggled(self, value: bool):
        """Show or hide the feature-selection checkboxes."""
        self.features_container.visible = value

    # -- Helpers -------------------------------------------------------------

    def _get_selected_features(self) -> list[str] | None:
        """
        Return the list of currently checked feature names to train on.

        Returns None (and sets a status message) if no feature is checked.
        """
        selected = [name for name, checkbox in self._feature_checkboxes.items() if checkbox.value]
        if not selected:
            self.status_label.value = "Select at least one training feature."
            return None
        return selected

    def _gather_training_triplets(self):
        """
        Collect (raw, mask, train_labels) triplets from every finalized image
        on disk, plus the current image's in-progress training layer.

        Returns None (and sets a status message) if there is nothing to train on.
        """
        if self.current_image_path is None:
            self.status_label.value = "Click Start first."
            return None

        triplets = find_saved_triplets(self.folders, skip=self.current_image_path)
        triplets.append(
            (self.raw_layer.data, self.mask_layer.data, self.training_layer.data)
        )
        return triplets

    def _load_image(self, image_path: Path):
        """Load an image and its mask into the viewer, resetting the prediction and training layers."""
        raw = skimage.io.imread(str(image_path))
        mask = ensure_instance_labels(
            skimage.io.imread(str(get_mask_path(image_path, self.folders)))
        )

        self.current_image_path = image_path

        for layer in (
            self.raw_layer, self.mask_layer, self.training_layer, self.prediction_layer
        ):
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)

        self.raw_layer = self.viewer.add_image(raw, name=f"raw: {image_path.name}")
        self.mask_layer = self.viewer.add_labels(mask, name=f"mask: {image_path.name}")
        self.mask_layer.editable = False
        self.mask_layer.contour = 1

        training_data = np.zeros_like(mask, dtype=np.uint8)
        self.training_layer = self.viewer.add_labels(training_data, name="training")

        # Added after the training layer so outlines stay visible over filled objects.
        self.prediction_layer = self.viewer.add_labels(
            np.zeros_like(mask, dtype=np.uint8), name="prediction"
        )
        self.prediction_layer.contour = 1
        self.prediction_layer.editable = False

    def _mask_value_at(self, event) -> int | None:
        """
        Return the object ID under the cursor, read straight from the mask
        data so it works regardless of which layer is selected or visible.
        """
        if self.viewer.dims.ndisplay == 3:
            return self.mask_layer.get_value(event.position, world=True)

        coord = np.round(self.mask_layer.world_to_data(event.position)).astype(int)
        data = self.mask_layer.data
        if coord.shape != (data.ndim,) or not all(
            0 <= c < s for c, s in zip(coord, data.shape)
        ):
            return None
        return int(data[tuple(coord)])

    def _assign_class_at_click(self, viewer, event):
        """
        When the user clicks an object in the image, look up its ID in the
        mask layer's data and assign that object's entire footprint in the
        training layer to the currently selected class.

        Bound at the viewer level and independent of which layer is selected
        or visible, so the prediction layer can stay active.

        If the object is already assigned to that same class, clicking it
        again clears the assignment (sets it back to 0/unlabeled) so
        mistakes can be corrected. Clicking an object that is assigned to a
        different class reassigns it to the newly selected class.
        """
        if self.mask_layer is None:
            return

        label_id = self._mask_value_at(event)
        if label_id is None or label_id == 0:
            return

        class_index = self.class_names.index(self.current_class_combo.value) + 1
        object_voxels = self.mask_layer.data == label_id
        current_value = self.training_layer.data[object_voxels]
        if np.all(current_value == class_index):
            new_value = 0
        else:
            new_value = class_index
        self.training_layer.data[object_voxels] = new_value
        self.training_layer.refresh()


def main():
    """Launch napari with the classifier training widget docked."""
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    try:
        widget_kwargs = {"config_path": config_path}
        # Fail fast with a clear message before opening any napari window.
        load_class_names(config_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    viewer = napari.Viewer()
    widget = ClassifierGuiWidget(viewer, **widget_kwargs)
    viewer.window.add_dock_widget(widget, name="Object Classifier Trainer")
    napari.run()


if __name__ == "__main__":
    main()
