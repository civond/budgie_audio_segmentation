<h1>Budgie Audio Segmentation</h1>
This is a RMS-Hilbert transform based event detection algorithm that operates on budgerigar vocalizations. It functions by detecting regions of sound, and then performs segmentation on each region based on the Hilbert envelope of the audio signal.

<img src="figure.png"><img>

<h2>Usage</h2>
Use piezoelectric audio in .flac format</br></br>

<div>
1. Clone this repository into your own local machine.

```git
git clone [path]
```
</div>
<div>
2. Install and create a miniconda environment and add it to path.</br> 

```console
conda create --name budgie 
```

</div>

<div>
2. To install dependencies the following command using pip in the terminal.

```console
pip install librosa toml pandas matplotlib
 ```
</div>

<div>
3. Change the audio path in the path section in spec_options.toml to correspond to the file you wish to segment.

```python
[paths]
piezo_audio_path = "data/my_audio_file.flac"
```
</div>

<div>
4. Run the script from the command line with segment_options.toml.

```console
python src/main.py segment_option.toml
```
</div>



