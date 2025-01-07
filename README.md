# Nested_PoE
## Setup

You can install and setup this repository as follows:

```
git clone https://github.com/VictoriaGraf/Nested_PoE
cd Nested_PoE
conda create -n npoe python=3.8.16
conda activate npoe
pip install -r requirements.txt
```

To perform experiments with diverse triggers, you will also need the OpenBackdoor repository.

```
git clone https://github.com/thunlp/OpenBackdoor
cp Nested_PoE/poison.py OpenBackdoor/
```

Finish the setup for OpenBackdoor including downloading clean datasets. Create poisoned data for your main dataset by editing appropriate configs in `OpenBackdoor/configs` and using `poison.py`. Then, run the following to create your formatted data:

```
python mix_triggers_4way.py
python make_bias-only.py
```

Repeat the creation of poisoned data with new triggers for pretraining. Then,

```
python make_bias-only_pretrain.py
python make_clean_small.py
```

An example script to test your setup and demonstrate how to use `npoe.py` is provided in `example_script.sh`.
