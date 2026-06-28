from manifest_common import manifest_cli

REFERRAL_KEYWORDS = {
    "akiec",
    "basal",
    "bcc",
    "carcinoma",
    "melanoma",
    "malignant",
    "scc",
    "squamous",
}


def label_for(class_name: str) -> str:
    normalized = class_name.lower().replace("-", "_").replace(" ", "_")
    return "referral" if any(keyword in normalized for keyword in REFERRAL_KEYWORDS) else "lower_signal"


if __name__ == "__main__":
    manifest_cli(
        description="Build a Vital binary skin-triage manifest from the Kaggle skin disease image dataset.",
        label_for=label_for,
        source="kaggle:ismailpromus/skin-diseases-image-dataset",
    )
