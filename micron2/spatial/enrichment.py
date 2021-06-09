"""
Spatial/group enrichment of cell types and marker intensity

Tests enrichment, and create summary figures
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, fisher_exact
import tqdm.auto as tqdm
import itertools

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rcParams
import seaborn as sns

def group_subgroup_enrich(group_vect, subgroup_vect, u_groups=None, u_subgroups=None):
  """ Test for subtype enrichment amongst members of a super-group

  Test this contingency table with a Fisher exact test:

                       in query group  |  in other group
                      -----------------|-----------------
     target subgroup |        A        |        B        |
  -------------------|-----------------|-----------------|
  reference subgroup |        C        |        D        |
                      -----------------|-----------------

  Args:
    group_vect (array-like): Labels shaped (N,) for N cells to test
    subgroup_vect (array-like): Labels same shape as `group_vect`
    u_groups (list-like): Unique values in `group_vect`. Default None (all unique values are tested)
    u_subtypes (list-like): Unique values in `subgroup_vect`. Default None (all unique values are tested)

  Returns:
    odds_df (pd.DataFrame): Odds ratios shaped (u_groups, u_subtypes)
    pval_df (pd.DataFrame): P-values shaped (u_groups, u_subtypes)

  """
  def get_ctable_counts(group, subgroup, query_group, target_subgroup, other_subgroup=None):
      A = np.sum((group==query_group) & (subgroup==target_subgroup))
      B = np.sum((group!=query_group) & (subgroup==target_subgroup))

      if other_subgroup is None:
        C = np.sum((group==query_group) & (subgroup!=target_subgroup))
        D = np.sum((group!=query_group) & (subgroup!=target_subgroup))
      else:
        C = np.sum((group==query_group) & (subgroup==other_subgroup))
        D = np.sum((group!=query_group) & (subgroup==other_subgroup))

      return A, B, C, D
      

  if u_groups is None:
    u_groups = np.unique(group_vect)
  
  if u_subgroups is None:
    u_subgroups = np.unique(subgroup_vect)

  odds_df = pd.DataFrame(index=u_groups, columns=u_subgroups, dtype=np.float32)
  pval_df = pd.DataFrame(index=u_groups, columns=u_subgroups, dtype=np.float32)

  np.random.seed(999)
  tot = len(u_groups)*len(u_subgroups)
  with tqdm.tqdm(itertools.product(u_groups, u_subgroups), total=tot) as pbar:
    for query_group, target_subgroup in pbar:
      A, B, C, D = get_ctable_counts(group_vect, subgroup_vect, query_group, target_subgroup)
      
      # ctable = pd.DataFrame(np.array([[A, B], [C,D]]), index=[target_subgroup, 'other subgroup'],
      #                       columns=[query_group+' group', 'other group'])
      ctable = np.array([[A, B], [C,D]])
      res = fisher_exact(ctable) 

      odds_df.loc[query_group, target_subgroup] = res[0]
      pval_df.loc[query_group, target_subgroup] = res[1]

  return odds_df, pval_df



def plot_group_subgroup_enrich(odds_df, pval_df, save=None):
  """ 
  Take the output from `group_subgroup_enrich` and plot it
  """

  # Fill zero odds-ratios with NA then take the log(odds)
  odf = odds_df.copy()
  odf[odf == 0] = np.min(odf[odf>0].values)
  odf = np.log10(odf)
  na_odf = pd.isna(odf)
  odf = odf.fillna(0)

  # Fill zero p-values with the nonzero min p-val then take the -log(pval)
  pdf = pval_df.copy()
  pdf_np = np.array(pdf.values)
  pdf[pdf == 0] = np.min(pdf_np[pdf_np>0])
  pdf = -np.log10(pdf)
  pdf[na_odf] = 1

  # Leave constant for now
  max_size = 0.45
  min_size = 0.05
  size_scale = np.linspace(min_size, max_size, 11)
  size_bins = np.linspace(0, np.max(pdf.values), 10)

  min_enrich = -2
  max_enrich = 2
  cmap = np.array(sns.color_palette('bwr', 51))
  enrich_bins = np.linspace(min_enrich, max_enrich, 50)

  plt.figure(figsize=(6, 7), dpi=180)
  ax = plt.gca()

  for i,s in enumerate(odds_df.columns):
      for j,f in enumerate(odds_df.index):
          p = np.digitize(pdf.loc[f,s], size_bins)
          e = np.digitize(odf.loc[f,s], enrich_bins)
          c = Circle((i,j), radius=size_scale[p], color=cmap[e], lw=0.5, ec='k')
          ax.add_artist(c)
          
  # legends
  pval_intervals = np.linspace(1, size_bins[-1], 4, dtype=np.int)
  for i,v in enumerate(pval_intervals):
      p = np.digitize(v, size_bins)
      coord = (len(odds_df.columns), i)
      c = Circle(coord, radius=size_scale[p], color='w', lw=0.5, ec='k')
      ax.add_artist(c)
      ax.annotate(f'1e-{v:2.0f}', coord, fontsize=4, ha='center', va='center')
      
  p = np.digitize(size_bins[-1], size_bins)
  for i,v in enumerate(np.linspace(min_enrich, max_enrich, 5)):
      e = np.digitize(v, enrich_bins)
      coord = (len(odds_df.columns)+1, i)
      c = Circle(coord, radius=size_scale[p], color=cmap[e], lw=0.5, ec='k')
      ax.annotate(f'{v:1.1f}', coord, fontsize=4, ha='center', va='center')
      ax.add_artist(c)
          
  ax.set_xlim([-1,len(odds_df.columns)+2])
  ax.set_ylim([-1,len(odds_df.index)])
  _ = ax.set_xticks(range(len(odds_df.columns)+2))
  _ = ax.set_yticks(range(len(odds_df.index)+1))
  _ = ax.set_xticklabels(list(odds_df.columns)+['-log10 pvalue', 'log10 odds'],rotation=90)
  _ = ax.set_yticklabels(list(odds_df.index)+[''],rotation=0)
  ax.set_aspect('equal')

  if save is not None:
    plt.savefig(save, bbox_inches='tight', transparent=True)



def marker_enrich(adata):
  cellgroups = np.unique(adata.obs.niche_labels)
  features = [f for f in adata.uns['channels'] if f!='DAPI']

  celltypes = adata.obs.celltype
  in_target = 'Epithelial_CDH'
  out_target = 'Epithelial'

  # stationary reference niche
  ref_niche = 'Epithelial'

  enrich = pd.DataFrame(index=cellgroups, columns=features, dtype=np.float32)
  pvals = pd.DataFrame(index=cellgroups, columns=features, dtype=np.float32)

  np.random.seed(999)
  for f in features:
    #fvals = np.log10(adata[:, f'{f}_membrane_mean'].X.toarray().flatten())
    fvals = adata[:, f'{f}_membrane_mean'].X.toarray().flatten()
    
    for s in cellgroups:
      sidx = (adata.obs.niche_labels == s) & (celltypes == in_target)
      #oidx = (adata.obs.niche_labels != s) & (celltypes == out_target)
      ## stationary reference niche
      oidx = (adata.obs.niche_labels == ref_niche) & (celltypes == out_target)
      
      if (np.sum(sidx)==0) or (np.sum(oidx)==0):
          enrich.loc[s,f] = 0
          pvals.loc[s,f] = 1
          continue
      
      svals = fvals[sidx]
      ovals = fvals[oidx]
      ovals = np.random.choice(ovals, len(svals), replace=False)

      smean = np.mean(svals)
      omean = np.mean(ovals)
      #fc = (smean - omean)/omean
      fc = smean / omean
      res = wilcoxon(svals, ovals)

      enrich.loc[s,f] = np.log2(fc)
      pvals.loc[s,f] = res[1]
