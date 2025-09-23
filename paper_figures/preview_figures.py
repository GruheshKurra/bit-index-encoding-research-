#!/usr/bin/env python3
"""
Quick preview script to display generated figures.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def preview_figures():
    """Display a grid of generated figures for quick preview."""
    figures_dir = Path(__file__).parent
    
    # List of figure files to preview
    figure_files = [
        'figure1_compression_comparison.png',
        'figure2_speed_performance.png', 
        'figure3_accuracy_analysis.png',
        'figure4_pareto_frontier.png',
        'figure5_scalability_analysis.png'
    ]
    
    # Create a large figure to display all subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    axes = axes.flatten()
    
    for i, filename in enumerate(figure_files):
        filepath = figures_dir / filename
        if filepath.exists():
            img = mpimg.imread(str(filepath))
            axes[i].imshow(img)
            axes[i].set_title(f"Figure {i+1}: {filename.replace('.png', '').replace('figure', '').replace('_', ' ').title()}")
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"Figure not found:\n{filename}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide the last subplot if we have an odd number of figures
    if len(figure_files) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.suptitle('BIE Research Paper - Generated Figures Overview', fontsize=20, y=0.98)
    plt.savefig(figures_dir / 'figures_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    preview_figures()