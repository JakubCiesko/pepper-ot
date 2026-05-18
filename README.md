# Scene-Aware Dialogue for the Pepper Robot Using Object Tracking and Large Language Models

The goal of this thesis is to design and implement a dialogue interface system
for natural communication with the Pepper robot, based on its visual perception
of its surroundings. The system extends object detection with object tracking,
scene memory, semantic scene modelling, and dialogue grounded in the current
visual context.

The implementation contains three main parts:

- `server/` - FastAPI server for perception, scene memory, captioning, and chat.
- `client/` - Pepper robot client written for the Python 2 NAOqi environment.
- `research/` - experiment, evaluation, and data-generation utilities used in
  the thesis.

## Python 3 Installation

Create and activate a local Python environment from the repository root:

```bash
python3 -m venv pepper-venv
source pepper-venv/bin/activate
python -m pip install --upgrade pip
```

Install the dependencies directly:

```bash
python -m pip install -r requirements.txt
```

Alternatively, install the repository as an editable Python package. This also
installs the dependencies from `requirements.txt` and exposes the `app` server
package and the `research` packages on `PYTHONPATH`:

```bash
python -m pip install -e .
```

Check with

```bash
pip show pepper_ot
```

The dependency list includes model, GPU, and research packages, so installation
can take a while and may require a CUDA-compatible environment for the full
pipeline.

## Documentation

The documentation source lives in `docs/` and is configured by `mkdocs.yml`.
After installing the Python dependencies, start the live local docs server from
the repository root:

```bash
mkdocs serve
```

By default, the site is served at:

```text
http://127.0.0.1:8000
```

To build static HTML documentation instead, run:

```bash
mkdocs build
```

The generated HTML output is written to `site/`.

## Running the Server

Run server commands from the `server/` directory because the application loads
`config.yaml` and static assets relative to that directory.

For local development without ngrok:

```bash
cd server
USE_NGROK=False uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server is then available at:

```text
http://127.0.0.1:8000
```

For a public ngrok tunnel, set your ngrok token and use the helper script:

```bash
cd server
export NGROK_AUTH_TOKEN=<your-ngrok-token>
bash start_server.sh
```

`server/start_server.sh` currently starts:

```bash
USE_NGROK=True uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

If you do not want development reload-on-change behavior, remove the `--reload`
flag from that command or run uvicorn directly without it:

```bash
cd server
USE_NGROK=False uvicorn app.main:app --host 0.0.0.0 --port 8000
```

When connecting the Pepper client, configure
`client/app/scripts/client_config.json` so `server.base_url` points either to
the local server URL or to the ngrok public URL.
Currently it is set to the public ngrok URL which is tied to our NGROK auth token.

## Pepper Client Setup

The Pepper client runs in the Python 2 NAOqi environment used by Pepper and
Choregraphe. Follow the FI MU Pepper installation instructions first:

https://nlp.fi.muni.cz/trac/pepper/wiki/InstallationInstructions

In short, the Pepper environment setup installs Choregraphe, a local Python 2
environment, the `pynaoqi` SDK, and the `PEPPER_ROOT` environment variable. The
instructions should leave Python 2 available at:

```bash
$PEPPER_ROOT/pyenv/bin/python2
```

Verify that the NAOqi Python bindings are available:

```bash
$PEPPER_ROOT/pyenv/bin/python2 -c 'import qi; print qi'
```

Then install the client dependencies from the repository root:

```bash
$PEPPER_ROOT/pyenv/bin/pip2 install -r client/requirements.txt
```

To run against a Choregraphe virtual robot:

1. Start the virtual robot in Choregraphe and note its port.
2. In `client/app/scripts/peppergroundedclient.py`, set `run_local = True`.
3. In the same file, switch `czech = True` or `czech = False` depending on the
   desired local dialogue language.
4. Start the service:

```bash
cd client
VIRTUAL_ROBOT_PORT=<PORT> ./start_service.sh
```

`client/start_service.sh` uses:

```bash
$PEPPER_ROOT/pyenv/bin/python2 app/scripts/peppergroundedclient.py --qi-url tcp://127.0.0.1:$VIRTUAL_ROBOT_PORT
```

After this, load project pml file in Choregraphe. After this and running the project on the virtual robot, you are able to communicate with the robot running the service in the dialog console.

### Pepper App Packaging and Deployment

`client/app/Makefile` provides convenience targets for packaging the Choregraphe
application as a `.pkg` file and installing it on a Pepper robot. These targets
assume the FI MU Pepper environment is already configured, including
`PEPPER_ROOT`.

To package the current application state:

```bash
cd client/app
make pkg
```

This uses `$PEPPER_ROOT/pyenv/bin/qipkg`, the local `.pml` file, `manifest.xml`,
and writes the generated package into `$PEPPER_ROOT/apps_dir`.

To install the latest generated package using the local FI MU Pepper tooling:

```bash
cd client/app
make install
```

This uses `$PEPPER_ROOT/apps_dir/install_pkg.py`, so it requires the same Pepper
environment and robot access setup as the standard FI MU installation workflow.

For fast client iteration against a physical robot, the Makefile also provides:

```bash
cd client/app
make deploy
```

`make deploy` first runs `make pkg`, then uploads the generated `.pkg` with
`scp` and runs a remote install command with `ssh`. It requires a reachable
intermediate machine from which the Pepper robot can be installed to, plus these
environment variables:

```bash
export FI_MUNI_USER=<your-fi-username>
export FI_THESIS_HOST=<remote-host>
export FI_THESIS_REMOTE_INSTALL_CMD=<remote-install-command>
export PEPPER_ROOT=<local-pepper-root>
```

The deploy target was added for fast development iteration: it packages the
current app state and installs it on the Pepper robot in one step.

For deployment to a physical Pepper robot, use the same Pepper/NAOqi Python 2
environment and configure the robot connection according to the Choregraphe and
FI MU Pepper setup instructions.

## Model Sources

The repository does not bundle third-party model weights. External model assets
are either downloaded by the runtime libraries when needed or documented here
with manual download instructions.

### RelTR

RelTR is used as an optional third-party scene-graph backend for learned visual
relation prediction. The RelTR architecture and checkpoint are not authored in
this thesis; this repository only contains integration code that runs RelTR from
the server scene-graph pipeline and maps its predictions onto tracked
detections.

- Upstream project: https://github.com/yrcong/RelTR
- Checkpoint used by the upstream authors:
  https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view

Download `checkpoint0149.pth` from the Google Drive link above and place it
under the server model directory:

```bash
mkdir -p THIS_REPO_ROOT/server/detection_models
mv ~/Downloads/checkpoint0149.pth THIS_REPO_ROOT/server/detection_models/reltr.pth
```

This matches the default RelTR checkpoint path in `server/config.yaml`:

```yaml
scene_graph:
  reltr:
    enabled: false
    checkpoint_path: detection_models/reltr.pth
```

RelTR is disabled by default. Enable it through `server/config.yaml` or through
the server dashboard only after the checkpoint is available locally.

The RelTR work should be cited as:

```text
Y. Cong, M. Y. Yang, and B. Rosenhahn. RelTR: Relation Transformer
for Scene Graph Generation. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 2023.
```

### PersonViT Re-Identification

Human re-identification uses the Hugging Face model:

https://huggingface.co/maennyn/personvit-reid-msmt17-vit-s

This model is a Hugging Face-compatible port/scaffold based on the original
PersonViT work:

https://github.com/hustvl/PersonViT

The original PersonViT project is distributed under the Apache License 2.0. The
server default is configured in `server/config.yaml` as:

```yaml
tracking:
  human_reid_model: "maennyn/personvit-reid-msmt17-vit-s"
```

No manual download or Git clone is needed for this model. It is downloaded by
the Transformers library when the tracking/re-identification component first
loads it with `AutoModel.from_pretrained(...)`.

Other model/provider assets used by Hugging Face, Roboflow, Ultralytics, and
similar libraries are also resolved by the corresponding runtime code when
needed. Do not include downloaded model caches, third-party checkpoints, or
upstream third-party repositories in the thesis submission archive; include only
the project source code, configs, results, and these download instructions.

## Data and Experiment Artifacts

The thesis experiments use a mixture of author-collected indoor images and
samples from the MIT Indoor Scene Recognition dataset:

https://web.mit.edu/torralba/www/indoor.html

To restore the MIT Indoor data locally, download `indoorCVPR_09.tar` from the
MIT page and extract it into `data/images`:

```bash
mkdir -p data/images
# Download indoorCVPR_09.tar from https://web.mit.edu/torralba/www/indoor.html
tar -xf indoorCVPR_09.tar -C data/images
```

After extraction, the expected dataset root is:

```text
data/images/Images/
```

The MIT Indoor Scene Recognition dataset contains 67 indoor categories and
15,620 JPG images. In this thesis, images were sampled from this dataset and
combined with author-collected images for the training and evaluation splits.
The dataset should be cited as:

```text
A. Quattoni and A. Torralba. Recognizing Indoor Scenes.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009.
```

The working training and evaluation folders used during the thesis were:

```text
data/images/train/
data/images/eval/
```

These folders may be empty in the submitted repository for space-saving
reasons. Partial author-collected image data may be included under:

```text
data/images/train_mine/
data/images/eval_mine/
```

The complete author-collected image data is available upon request at:

```text
jakub.ciesko@gmail.com
```

Experiment configurations and result files are available under:

```text
research/artifacts/experiments/
```

Human evaluation materials are available under:

```text
research/artifacts/human_eval/
```

Generated SoM image folders named `som_images_draft` inside experiment and run
directories may be removed from the public submission for storage reasons. The
remaining non-image experiment configs, metrics, reports, and human evaluation
artifacts are kept where possible.

## Background References

- Grassi, Lucrezia, et al. "Grounding conversational robots on vision through
  dense captioning and large language models." 2024 IEEE International
  Conference on Robotics and Automation (ICRA). IEEE, 2024.
- Sun, Jiangeng. "Intelligible dialogue manager for social robots: An AI
  dialogue robot solution based on Rasa open-source framework and Pepper robot."
  (2023).
- Abdel Hafez, Raneem. "Enhancing Human-Robot Interaction: Integrating Large
  Language Models and Advanced Speech Recognition into the Pepper Robot."
  (2024).
- Mascaro, Ruben, and Margarita Chli. "Scene representations for robotic spatial
  perception." Annual Review of Control, Robotics, and Autonomous Systems 8.1
  (2025): 351-377.
