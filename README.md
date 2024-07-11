# Threshold_Extract
In water index calculation, gets the best threshold and full area of water according to known parts or approximation of water. Landsat 5/7/8/9 is supported.
## Usage
### 1. Clone the repository.
    git clone https://github.com/Galasnow/Threshold_Extract.git

### 2. Install requirements. Conda env is recommended.
    cd Threshold_Extract
    conda create -n <name> python=3.10
    conda activate <name>
    conda install -c conda-forge gdal
    pip install -r requirements.txt

### 3. Download a package of Landsat 5/7/8/9 and unzip it to a folder.

### 4. Edit `config.yml` according to your setting.

### 5. Run the script.
    python src/threshold_extract.py config/config.yml

## Note
This is an experimental project and it can't guarantee any functionality now.
