from pipeline.new_patient_segmentation import BRATS_VALID_OUTPUT_LABELS


def test_installed_brats_model_output_label_contract() -> None:
    assert BRATS_VALID_OUTPUT_LABELS == {0, 1, 2, 4}
    assert 3 not in BRATS_VALID_OUTPUT_LABELS
