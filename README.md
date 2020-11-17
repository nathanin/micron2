# Micron 2 :microscope:

**< colorful picture goes here >**

*****  

- [x] [Segmentation](#segmentation) (via StarDist) :watermelon:
- [x] [Single cell clustering](#single-cell-clustering)
- [ ] Niche detection
- [ ] Spatial statistics
- [ ] Graph analysis


******

With paired scSeq :test_tube::dna:
- [ ] Spatial constraints on interaction analysis (imaging --> scSeq)
- [ ] Interacting cells co-occurance frequency (scSeq --> imaging)

See [snippets](#snippets) for usage.

*****

## Sub-goals

### Data class
- [x] store processed image data + nuclear masks + coordinates in hdf5
- [ ] short term: wrapper to use AnnData and store a hook to an open cell image dataset
- [ ] long term: extend the AnnData class

### Segmentation
- [x] pull data from images and perform statistics on these data quickly
- [ ] data loader for segmented data focusing on cells, tracking location and cell_ids

### Single cell clustering
- [x] cluster with normalized intensity values
- [ ] cluster with morphology
- [ ] cluster with morphology + staining


*****
## Environment

Docker: `rapidsai/rapidsai:0.16-cuda10.1-runtime-ubuntu16.04-py3.8`

Note: to use leidenlag install the proper igraph package from pip: `pip install python-igraph`


*****
## Snippets

Build a cell image dataset:
```python
x
```

Attach a cell image dataset to an AnnData object:
```python
x
```

Run unsupervised clustering on the cell images:
```python
x
```