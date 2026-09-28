# Local Face Recognition Pipeline

[![CI](https://github.com/joshuajyi/local-face-recognition-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/joshuajyi/local-face-recognition-pipeline/actions/workflows/ci.yml)

This is a local face recognition project I built to understand the complete
path from an input image to a final identity decision. I kept detection,
alignment, embedding, storage, and matching as separate parts instead of using
one library call that hides the process.

The pipeline can:

- Enroll a person from several reference photos
- Recognize faces in a saved photo or live webcam feed
- Reject a match as `Unknown` when its score is below the threshold
- Display facial landmarks, cosine similarity, stage timing, and estimated FPS
- Benchmark detection, embedding, matching, and total processing time
- Evaluate a labeled test set and report recognition errors

All face photos, embeddings, models, and generated results stay on the local
computer and are excluded from Git.

## Demo

![Live webcam recognition demo](docs/demo.jpg)

This demo ran on my 2024 M4 MacBook Pro. YuNet detected the face and returned
five landmarks, SFace created the embedding, and cosine similarity matched it
to the enrolled profile.

## Quick start

Clone the repository and enter the project folder:

```bash
git clone https://github.com/joshuajyi/local-face-recognition-pipeline.git
cd local-face-recognition-pipeline
```

Install `uv` if needed, then create the locked development environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev
```

Download the versioned YuNet and SFace model files:

```bash
uv run face-pipeline download-models
```

The download command verifies each model with SHA-256 before it is used.

## Enroll a profile

Create private folders for enrollment and testing:

```bash
mkdir -p private_photos/enroll private_photos/test
```

Add several clear photos of one person to `private_photos/enroll/`. Different
angles, lighting, and glasses make the saved profile more representative. Keep
separate photos for testing so the system is not evaluated on its enrollment
images.

Enroll the images under one name:

```bash
uv run face-pipeline enroll "Joshua" private_photos/enroll/*.jpg
```

The command detects the largest face in each photo, aligns it, creates a
normalized embedding, averages the embeddings, and saves one local profile.

List the enrolled profiles:

```bash
uv run face-pipeline profiles
```

Profiles are stored in `data/gallery.npz`. Running enrollment again with the
same name replaces the older profile.

## Recognize a photo

```bash
uv run face-pipeline recognize \
  --image private_photos/test/joshua_test.jpg \
  --show
```

The terminal prints the predicted label, similarity score, detection
confidence, and timing. An annotated copy is saved in `outputs/`.

To test rejection, use a consenting person who was not enrolled:

```bash
uv run face-pipeline recognize \
  --image private_photos/test/unknown_test.jpg \
  --show
```

## Run the webcam

```bash
uv run face-pipeline recognize --camera 0
```

Press `q` or Escape to close the window. On macOS, Terminal or the editor used
to run the command may need camera permission under **System Settings > Privacy
& Security > Camera**.

## Evaluate recognition results

The evaluation command makes testing repeatable instead of relying on one good
demo image. Organize single-face test images by their expected label:

```text
private_photos/evaluation/
  Joshua/
    front.jpg
    glasses.jpg
  Unknown/
    person_1.jpg
    person_2.jpg
```

Known folder names must match the enrolled profile names. Use `Unknown` for
people who are not enrolled. Each evaluation image should contain exactly one
face.

Run the evaluation:

```bash
uv run face-pipeline evaluate private_photos/evaluation
```

The report includes accuracy over usable images, false accepts, false rejects,
wrong identities, images with no detected face, and images with multiple faces.
Full per-image results are saved to `outputs/evaluation.json`.

The error types matter differently:

- A **false accept** means an unknown person was labeled as an enrolled person.
- A **false reject** means an enrolled person was labeled as unknown.
- A **wrong identity** means one enrolled person was labeled as another.

## Benchmark performance

```bash
uv run face-pipeline benchmark \
  private_photos/test/joshua_test.jpg \
  private_photos/test/unknown_test.jpg \
  --repeat 10
```

The benchmark warms the models first, then records median and p95 times for
each stage. System information, configuration, model hashes, and complete
results are saved to `outputs/benchmark.json`.

## How the pipeline works

1. **Detection:** YuNet finds each face and returns a box, five landmarks, and
   a confidence score.
2. **Alignment:** SFace uses the landmarks to rotate and crop the face into a
   consistent position.
3. **Embedding:** SFace converts the aligned face into a numeric vector.
4. **Enrollment:** Reference vectors are normalized, averaged, and normalized
   again to create one profile.
5. **Matching:** A query vector is compared with every profile using cosine
   similarity.
6. **Decision:** The best match is accepted only when its score passes the
   configured threshold. Otherwise, the result is `Unknown`.

The default cosine threshold is `0.363`, based on OpenCV's SFace example. A
higher value is stricter. It may reduce false accepts while causing more false
rejects, which is why the evaluation command reports both.

## Tools and design choices

- **Python 3.11** keeps the project easy to run and test on a laptop.
- **OpenCV** loads the models, handles images and webcam frames, aligns faces,
  and draws the output.
- **YuNet** is a small face detector that also returns five landmarks.
- **SFace** creates the face embeddings used for recognition.
- **ONNX** lets OpenCV run both models locally without a cloud API.
- **NumPy** handles normalization, profile averaging, similarity, and gallery
  storage.
- **uv** recreates the environment from a committed lockfile.
- **pytest** tests the matching, gallery, and evaluation logic without needing
  private photos or a webcam.
- **Ruff** catches common Python mistakes and keeps imports consistent.
- **GitHub Actions** runs the locked install, linter, and tests on every push
  and pull request.

I considered using DeepFace, but it combines most of the pipeline behind one
interface. YuNet and SFace required more setup, but they let me inspect and
test each stage separately.

## Results from my laptop

I enrolled one profile from six photos, then tested one separate photo of the
same person and one consenting person who was not enrolled. The benchmark ran
both test images 10 times each.

| Measurement | Result |
|---|---:|
| Computer | 2024 MacBook Pro, M4, 16 GB RAM |
| Test image size | 600 x 800 |
| Detection median | 4.75 ms |
| Embedding median | 5.67 ms |
| End-to-end median | 10.58 ms |
| End-to-end p95 | 13.48 ms |
| Pipeline FPS from median | 94.55 FPS |
| Known test photos accepted | 1/1 |
| Unknown test photos rejected | 1/1 |

This is a small personal test, not a full accuracy study. The FPS value comes
from saved images and should not be treated as the exact webcam frame rate.

## Testing and Git workflow

Run the same checks used by GitHub Actions:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The unit tests cover vector normalization, cosine similarity, profile storage,
nearest-profile matching, unknown rejection, evaluation labels, error types,
and summary metrics. Model files and private biometric data are not needed for
the test suite.

I built the repository in separate commits so the history shows the project
moving from vector matching, to model inference, to CLI commands, tests,
benchmarking, and evaluation. Before each commit I use `git status` and
`git diff` to verify what will be published.

## Project structure

```text
.github/workflows/ci.yml  Automated lint and test checks
src/face_pipeline/
  cli.py                  Command-line interface
  config.py               Paths and thresholds
  download.py             Model downloads and checksum verification
  evaluation.py           Labeled test discovery and error metrics
  gallery.py              Local profile storage and nearest match
  matching.py             Embedding normalization and cosine similarity
  pipeline.py             End-to-end processing, timing, and drawing
  vision.py               YuNet and SFace wrappers
tests/                     Unit tests that do not need private data
docs/                      Public demo image
```

## A problem I ran into

YuNet did not reliably detect my face in the original 1536 x 2048 phone
photos. Resizing the same photos to 600 x 800 fixed the issue. That was a good
reminder that input preprocessing can change a model's result even when the
model and recognition code stay the same.

## Privacy and limitations

- Face photos and embeddings are sensitive biometric data. This project keeps
  them local and excludes them from Git.
- The threshold has not been calibrated on a large or diverse dataset.
- Lighting, pose, blur, glasses, and camera distance can change the score.
- There is no liveness detection, so a photo shown to the camera may fool it.
- The gallery uses a linear search and is intended for a small number of
  profiles.
- This is an educational prototype, not an authentication or surveillance
  product.

## Development note

I used ChatGPT/Codex to help review early code, debug setup problems, and check
documentation. I ran the project locally, tested it with my own controlled
images, and verified how data moves through every stage. Recognition itself is
fully local and does not call an online face recognition service.

## References

- [OpenCV face detection and recognition tutorial](https://docs.opencv.org/5.0/tutorials/dnn/dnn_face/dnn_face.html)
- [YuNet model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [SFace model](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
