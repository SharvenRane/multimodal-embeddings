# multimodal-embeddings

A small, self contained study of joint multimodal embeddings. The idea is the
one behind CLIP: take two views of the same thing, push each through its own
encoder, and train so that the two views of a single example land near each
other in a shared space while unrelated examples land far apart. Here the two
views stand in for something like an image and a tabular record, or an image
and a text vector.

Everything runs on CPU in a couple of seconds. There are no downloads and no
pretrained weights. The data is synthetic but it is built so that the alignment
problem is real rather than trivial.

## How the data is made

Each example starts from a shared latent vector. That latent is passed through
two fixed random linear maps, one per view, and a bit of view specific noise is
added on top. So a matched pair genuinely carries common signal, while two
randomly chosen examples share nothing but noise. A contrastive model has to
recover that common factor across the two different view spaces, which is the
whole point of the exercise. See `src/data.py`.

## The model

Two projection heads, one per view, each a tiny MLP that maps its raw view into
a shared embedding space. Outputs are L2 normalised, so cosine similarity is
just a dot product. The heads do not share weights because the two views live
in different spaces. See `src/model.py`.

## The objective

Symmetric InfoNCE, the same contrastive loss CLIP uses. For a batch, you form
the similarity matrix between the two sets of embeddings. The diagonal holds the
matched pairs and everything off the diagonal is a mismatched pair. The loss
pulls the diagonal up and pushes the rest down, in both directions. See
`src/losses.py`.

## Running it

Install the requirements, then:

```python
from src.data import make_paired_data
from src.train import train_aligner, mean_matched_minus_mismatched

view_a, view_b = make_paired_data(n=512, dim_a=64, dim_b=48, noise=0.3)
result = train_aligner(view_a, view_b, epochs=30)

gap = mean_matched_minus_mismatched(result.model, view_a, view_b)
print("matched minus mismatched cosine gap:", gap)
```

The `gap_history` on the result records the alignment gap at every epoch, so you
can watch matched pairs pull ahead of random pairs as training proceeds.

## Tests

The tests check behaviour rather than fixed numbers. They confirm that
projection outputs are unit norm, that the contrastive loss is lower when
embeddings are aligned than when they are shuffled, that matched pairs end up
with higher cosine similarity than mismatched ones after training, that the
alignment gap grows over training, and that each view's true partner tends to
be its nearest neighbour in the other view.

```
python -m pytest tests/ -q
```

On the reference run all of them pass.
