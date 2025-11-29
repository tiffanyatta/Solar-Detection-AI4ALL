"""
Script to download the Kaggle dataset for the Solar Detection project.
"""

import os
import shutil
import kagglehub

def download_dataset():
    """Download the dataset from Kaggle and move it to the data directory."""
    
    print("📥 Downloading dataset from Kaggle...")
    print("This may take a few minutes...")
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download("saurabhshahane/northern-hemisphere-horizontal-photovoltaic")
        
        print(f"✅ Dataset downloaded to: {path}")
        print("📁 Looking for CSV files...")
        
        # Find CSV files in the downloaded directory
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            print("❌ No CSV files found in the downloaded dataset.")
            print(f"📂 Please check the directory: {path}")
            return False
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Copy the first CSV file (or all if there are multiple)
        if len(csv_files) == 1:
            # Single CSV file - copy it as solar_data.csv
            source_file = csv_files[0]
            dest_file = os.path.join("data", "solar_data.csv")
            shutil.copy2(source_file, dest_file)
            print(f"✅ Copied {os.path.basename(source_file)} to data/solar_data.csv")
        else:
            # Multiple CSV files - copy all and name the first one as solar_data.csv
            print(f"📄 Found {len(csv_files)} CSV files:")
            for i, csv_file in enumerate(csv_files, 1):
                filename = os.path.basename(csv_file)
                print(f"   {i}. {filename}")
            
            # Copy the first one as solar_data.csv
            source_file = csv_files[0]
            dest_file = os.path.join("data", "solar_data.csv")
            shutil.copy2(source_file, dest_file)
            print(f"\n✅ Copied {os.path.basename(source_file)} to data/solar_data.csv")
            
            # Copy others with their original names
            for csv_file in csv_files[1:]:
                filename = os.path.basename(csv_file)
                dest_file = os.path.join("data", filename)
                shutil.copy2(csv_file, dest_file)
                print(f"✅ Also copied {filename} to data/")
        
        print("\n🎉 Dataset ready! You can now run the Streamlit app.")
        print("   Run: streamlit run app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic")
        print("   And save it as data/solar_data.csv")
        return False


if __name__ == "__main__":
    download_dataset()

