# Micron 2 :microscope:

**< colorful picture goes here >**

*****  

- [x] [Segmentation](#segmentation) (via StarDist) :watermelon:
- [ ] [Single cell clustering](#single-cell-clustering)
- [ ] Niche detection
- [ ] Spatial statistics
- [ ] Graph analysis


******

With paired scSeq :test_tube::dna:
- [ ] Spatial constraints on interaction analysis (imaging --> scSeq)
- [ ] Interacting cells co-occurance frequency (scSeq --> imaging)

*****

## Sub-goals

### Segmentation
- [X] pull data from images and perform statistics on these data quickly
- [ ] data loader for segmented data focusing on cells, tracking location and cell_ids

### Single cell clustering
- [X] cluster with normalized intensity values
- [ ] cluster with morphology
- [ ] cluster with morphology + staining


*****
#### Environment

Docker: `rapidsai/rapidsai:0.16-cuda10.1-runtime-ubuntu16.04-py3.8`
