# Current VIGOR entry point

Use the [root README](../README.md) for the supported corrected E2 environment, four fresh15 commands, fixed model14 evaluation, checkpoint identities and metrics. [Data preparation](../docs/data-preparation.md) includes the pinned sky-mask/depth chain.

The compatibility default is `legacy`; the current recipe explicitly selects `--model_behavior corrected --translation_origin crop`. State dictionaries do not encode these switches or Gaussian ranges. Public evaluation rejects incomplete/incompatible checkpoints.

Model-only snapshots do not restore optimizer/RNG state exactly. Historical `--resume` and `--continue_weights` are weights-continuation paths, not the current continuous fresh15 recipe. Human epoch = saved index + 1; validation is separate from official test.
