"""
Fig 2B: batch-effect ARI (KMeans on UMAP must recover the experiment day; lower is better).

Same computation as fig2c_ari.py, run with --target_label experiment --batch_mode False.
    python experiments/resnet50/fig2b_ari.py [--quick]
"""
import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], '--target_label', 'experiment', '--batch_mode', 'False', *sys.argv[1:]]
runpy.run_path(str(Path(__file__).with_name('fig2c_ari.py')), run_name='__main__')
