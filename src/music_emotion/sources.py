from dataclasses import dataclass


@dataclass(frozen=True)
class Artifact:
    dataset: str
    name: str
    url: str
    extract_to: str
    note: str


PHASE_0_2_ARTIFACTS = [
    Artifact(
        dataset="emopia",
        name="EMOPIA_2.2.zip",
        url="https://zenodo.org/api/records/5257995/files/EMOPIA_2.2.zip/content",
        extract_to="emopia",
        note="Latest EMOPIA symbolic archive.",
    ),
    Artifact(
        dataset="deam",
        name="metadata.zip",
        url="https://cvml.unige.ch/databases/DEAM/metadata.zip",
        extract_to="deam",
        note="DEAM metadata.",
    ),
    Artifact(
        dataset="deam",
        name="DEAM_Annotations.zip",
        url="https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip",
        extract_to="deam",
        note="DEAM valence and arousal labels.",
    ),
    Artifact(
        dataset="deam",
        name="features.zip",
        url="https://cvml.unige.ch/databases/DEAM/features.zip",
        extract_to="deam",
        note="DEAM openSMILE audio features.",
    ),
    Artifact(
        dataset="deam",
        name="DEAM_audio.zip",
        url="https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip",
        extract_to="deam",
        note="DEAM Creative Commons audio.",
    ),
    Artifact(
        dataset="merge",
        name="MERGE_Lyrics_Complete.zip",
        url="https://zenodo.org/api/records/13939205/files/MERGE_Lyrics_Complete.zip/content",
        extract_to="merge",
        note="MERGE complete lyrics archive.",
    ),
    Artifact(
        dataset="merge",
        name="MERGE_Audio_Complete.zip",
        url="https://zenodo.org/api/records/13939205/files/MERGE_Audio_Complete.zip/content",
        extract_to="merge",
        note="MERGE complete audio archive.",
    ),
    Artifact(
        dataset="merge",
        name="MERGE_Bimodal_Complete.zip",
        url="https://zenodo.org/api/records/13939205/files/MERGE_Bimodal_Complete.zip/content",
        extract_to="merge",
        note="MERGE complete paired audio and lyrics archive.",
    ),
]


BLOCKED_ARTIFACTS = [
    Artifact(
        dataset="lira",
        name="LIRA_3.zip",
        url="https://data.mendeley.com/public-api/zip/9zdww6wnyx/download/3",
        extract_to="lira",
        note="Public page is open, but its archive endpoint rejects automated download.",
    ),
]
