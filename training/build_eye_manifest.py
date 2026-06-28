from manifest_common import manifest_cli


def label_for(class_name: str) -> str:
    return "lower_signal" if class_name.lower() == "normal" else "referral"


if __name__ == "__main__":
    manifest_cli(
        description="Build a Vital binary eye-triage manifest from the Kaggle eye disease classification dataset.",
        label_for=label_for,
        source="kaggle:gunavenkatdoddi/eye-diseases-classification",
    )
