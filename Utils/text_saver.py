"""
Simple text saver module for saving labeling algorithm results.
"""

def save_results_to_txt(results, filename="results.txt"):
    """
    Save results to a text file without any conversion.
    
    Args:
        results: List of result dictionaries from labeling algorithm
        filename: Output filename (default: results.txt)
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Total Results: {len(results)}\n")
        f.write("="*70 + "\n\n")
        
        for idx, res in enumerate(results, 1):
            f.write(f"Result #{idx}\n")
            f.write("-"*70 + "\n")
            
            # Write all keys and values as-is
            for key, value in res.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write(f"End of Results ({len(results)} total)\n")
    
    print(f"✅ Results saved to: {filename}")
