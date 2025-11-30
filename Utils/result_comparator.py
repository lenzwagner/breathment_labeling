"""
Result file comparator module.
Compares two result.txt files to check if they are identical.
"""

def compare_result_files(file1, file2):
    """
    Compare two result files and check if they are identical.
    
    Args:
        file1: Path to first result file
        file2: Path to second result file
    
    Returns:
        bool: True if files are identical, False otherwise
    """
    try:
        # Read both files
        with open(file1, 'r', encoding='utf-8') as f1:
            content1 = f1.read()
        
        with open(file2, 'r', encoding='utf-8') as f2:
            content2 = f2.read()
        
        # Compare contents
        if content1 == content2:
            print(f"\n{'='*70}")
            print("RESULT FILE COMPARISON")
            print(f"{'='*70}")
            print(f"✅ Files are IDENTICAL!")
            print(f"   File 1: {file1}")
            print(f"   File 2: {file2}")
            print(f"{'='*70}\n")
            return True
        else:
            print(f"\n{'='*70}")
            print("RESULT FILE COMPARISON")
            print(f"{'='*70}")
            print(f"❌ Files are DIFFERENT!")
            print(f"   File 1: {file1} ({len(content1)} chars)")
            print(f"   File 2: {file2} ({len(content2)} chars)")
            
            # Find first difference
            min_len = min(len(content1), len(content2))
            for i in range(min_len):
                if content1[i] != content2[i]:
                    print(f"   First difference at position {i}")
                    print(f"   Context: ...{content1[max(0,i-20):i+20]}...")
                    break
            
            if len(content1) != len(content2):
                print(f"   Length difference: {abs(len(content1) - len(content2))} chars")
            
            print(f"{'='*70}\n")
            return False
            
    except FileNotFoundError as e:
        print(f"\n{'='*70}")
        print("RESULT FILE COMPARISON - ERROR")
        print(f"{'='*70}")
        print(f"❌ File not found: {e}")
        print(f"{'='*70}\n")
        return False
    except Exception as e:
        print(f"\n{'='*70}")
        print("RESULT FILE COMPARISON - ERROR")
        print(f"{'='*70}")
        print(f"❌ Error comparing files: {e}")
        print(f"{'='*70}\n")
        return False
