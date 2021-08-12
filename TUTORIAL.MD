<h2>MUCS 2021: MUltilingual and Code-Switching ASR Challenges for Low Resource Indian Languages with espnet</h2>

Details about the challenge can be found <a href='https://navana-tech.github.io/MUCS2021/'>here</a>. This tutorial will cover building ASR models with ESPnet, for the  2 subtasks present in the challenge. <a href='https://github.com/espnet/espnet'>ESPnet</a> is an end-to-end speech processing toolkit which supports various State-of-the-art ASR, TTS models.

To use ESPnet on a new dataset, a recipe can be built and incorporated with ESPnet codebase. The recipe be present in two styles:
- espnet1 recipes: It follows kaldi style recipe. These recipies are located at ```espnet/egs```.
- espnet2 recipes: Independent of kaldi, allows for certain benifits such as common template. These recipies are located at ```espnet/egs2```.

I will be going through steps involved in both the approaches for mucs challenge subtasks. To start with, ESPnet has to be installed with kaldi support (kaldi may be necessary for certian preprocessing steps in espnet2).

<h4>Installing ESPnet</h4>
All steps can be found <a href='https://espnet.github.io/espnet/installation.html'>here</a>, I will be going key steps.

1. Clone espnet repository
```
git clone https://github.com/espnet/espnet
```
2. To install kaldi, follow the steps mentioned <a href='https://kaldi-asr.org/doc/install.html'>here</a>. After installing kaldi, create a symbolic link to kaldi inside ESPnet. With this symbolic link, kaldi functionalities can be directly utilised in ESPnet.
```
cd <espnet-root>/tools
ln -s <kaldi-root> .
```
3. To specify Python intepreter to be used for ESPnet, ```<espnet-root>/tools/activate_python.sh``` has to be created. If you are not operating inside a virtual envoronemnt, then you can place an empty file as shown below.
```
cd <espnet-root>/tools
rm -f activate_python.sh && touch activate_python.sh
```
If you are using venv or an anaconda envirornment, follow the steps mentioned in the full installation documentation.
4. To install ESPnet, execute the make file.
```
cd <espnet-root>/tools
make
```
This will install all dependencies. If installation fails on any particular package, it can be due to the default version supported by ESPnet is not compatible with your system. In that case, either upgrade your package dependecies (along with CUDA, Cudnn version if that is relevent) or specify a different version to be installed with ESPnet. For example, with pytorch:  
```
cd <espnet-root>/tools
make TH_VERSION=1.7.0 CUDA_VERSION=10.1
```

ESPnet is ready to use.

<h4>Working with data</h4>

When you are done acquiring your data, you will have a set of audio files in a folder, such as
<root-data-folder>
```
-<root-data-folder>
  -001050238.wav
  -002100128.wav
  -001820055.wav
  ...
```
You will also have it's corresponding transcripts. It can be present in a text file as shown below
 ```
001050238	ఈ సందర్భంగా చంద్రబాబు అమరావతి నగర అభివృద్ధిపై తన ఆలోచనలను వారికి వివరించారు
002100128	ఓహియో అయోవా పెన్సిల్వేనియా రాష్ట్రాల్లో ఒబామా అధిక్యం కొనసాగుతోంది
001820055	చంద్రబాబు అందరికీ క్షమాపణ చెప్పాలని ఆయన రాజీనామా చేసే వరకు వదలకూడదన్నారు
...
 ```
In the text file, the unique file names are present along with the text next to it. This is how data is present for mucs subtask1, I will walk through the data preparation steps involved with it.

<h4>Subtask1 ESPnet recipe</h4>
Subtask1 involves building a multilingual ASR system in six languages, namely, Hindi, Marathi, Odia, Telugu, Tamil, and Gujarati. 3 languages- Hindi, Marathi, Odia are available at a publicly accesible download link, so let us have a script to download it for us.

The ```<root-data-folder>``` is defined by the ```path``` variable which can be passed as an arguement while running the script. We then store the full path to the directory in ```DIR```.
```
path=$1
cwd=`pwd`
DIR=$cwd/$path
```
Now, we will define an associative arrays for our links. This will act link a python dictionary allowing easy access. Here ```<link>``` corresponds to the download links for each dataset.
```
declare -A trainset
trainset['Hindi']=<link>
trainset['Marathi']=<link>
trainset['Odia']=<link>

declare -A testset
testset['Hindi']=<link>
testset['Marathi']=<link>
testset['Odia']=<link>
```
Now that we have the links defined, we can traverse through it, download it & unzip the files. We will create a new folder for each langauge inside '''DIR''' & path ```.done``` files so that there codes won't be executed again once the data has been downloaded.
```
for lang in Hindi Marathi Odia; do
  if [ ! -e ${DIR}/${lang}.done ]; then
      cd ${DIR}
      mkdir -p ${lang}
      cd ${lang}
      wget -O test.zip ${testset[$lang]}
      tar xf "test.zip"
      rm test.zip
      wget -O train.zip ${trainset[$lang]}
      tar xf "train.zip"
      rm train.zip
      cd $cwd
      echo "Successfully finished downloading $lang data."
      touch ${DIR}/${lang}.done
  else
      echo "$lang data already exists. Skip download."
  fi
done
```

We will place all the codes inside ```download_data.sh``` so that it becomes
```
path=$1
cwd=`pwd`
DIR=$cwd/$path

declare -A trainset
trainset['Hindi']=<link>
trainset['Marathi']=<link>
trainset['Odia']=<link>

declare -A testset
testset['Hindi']=<link>
testset['Marathi']=<link>
testset['Odia']=<link>

for lang in Hindi Marathi Odia; do
  if [ ! -e ${DIR}/${lang}.done ]; then
      cd ${DIR}
      mkdir -p ${lang}
      cd ${lang}
      wget -O test.zip ${testset[$lang]}
      tar xf "test.zip"
      rm test.zip
      wget -O train.zip ${trainset[$lang]}
      tar xf "train.zip"
      rm train.zip
      cd $cwd
      echo "Successfully finished downloading $lang data."
      touch ${DIR}/${lang}.done
  else
      echo "$lang data already exists. Skip download."
  fi
done
```
After running this code, we will have the following structure,
```
-<root-data-folder>
    -Hindi/
        -train/
            -audio/
                -001050238.wav
                -002100128.wav
                -001820055.wav
                ...
            -transcripts.txt
        -test/
            -audio/
            -transcripts.txt
    -Marathi/
        -train/
            -audio/
            -transcripts.txt
        -test/
            -audio/
            -transcripts.txt
    -Odia/
        -train/
            -audio/
            -transcripts.txt
        -test/
            -audio/
            -transcripts.txt

```
Next, we need to add data from Telugu, Tamil, and Gujarati which can be accessed with a login. So we first need to manually download the data and place it in ```<root-data-folder>```. In this data, we find that the data is present as
```
-<lang folder>
    -$lang-in-Test/
        -Audios/
        -transcripts.txt
    -$lang-in-Train/
        -Audios/
        -transcripts.txt
```
We need to restructure it to match the structure of the data we stored previously. Let us define associative arrays as follows,
```
path=$1
cwd=`pwd`
DIR=$cwd/$path

declare -A msrdata_train
msrdata_train['Tamil']=ta-in-Train
msrdata_train['Telugu']=te-in-Train
msrdata_train['Gujarati']=gu-in-Train

declare -A msrdata_test
msrdata_test['Tamil']=ta-in-Test
msrdata_test['Telugu']=te-in-Test
msrdata_test['Gujarati']=gu-in-Test
```
Now we can traverse through it as we have done previously,
```
for lang in Tamil Telugu Gujarati; do
  if [ ! -e ${DIR}/${lang}.done ]; then
      cd ${DIR}
      mkdir -p ${lang}
      cd ${lang}
      cp -r ../microsoftspeechcorpusindianlanguages/${msrdata_train[$lang]} train
      cp -r ../microsoftspeechcorpusindianlanguages/${msrdata_test[$lang]} test
      mkdir train/audio
      mkdir test/audio
```
The audio files in this dataset are present at 16Khz while the other 3 langauge audios are present at 8Khz, so we need to downsample this data to 8Khz as well.
```
      down_sample.sh "$DIR/$lang/train/Audios/*" $DIR/$lang/train/'audio'/
      down_sample.sh "$DIR/$lang/test/Audios/*" $DIR/$lang/test/'audio'/
```
Where ```down_sample.sh``` is as follows,
```
DIR=$1
reDir=$2

for i in $DIR; do
    ffmpeg -y  -i "$i" -ar 8000 "$reDir${i##*/}"
done
```
Here, we traverse through each file and downsample it and store it in a different folder. Finally, we can combine all the codes and place it in ```prepare_data.sh```
```
declare -A msrdata_train
msrdata_train['Tamil']=ta-in-Train
msrdata_train['Telugu']=te-in-Train
msrdata_train['Gujarati']=gu-in-Train

declare -A msrdata_test
msrdata_test['Tamil']=ta-in-Test
msrdata_test['Telugu']=te-in-Test
msrdata_test['Gujarati']=gu-in-Test

for lang in Tamil Telugu Gujarati; do
  if [ ! -e ${DIR}/${lang}.done ]; then
      cd ${DIR}
      mkdir -p ${lang}
      cd ${lang}
      cp -r ../microsoftspeechcorpusindianlanguages/${msrdata_train[$lang]} train
      cp -r ../microsoftspeechcorpusindianlanguages/${msrdata_test[$lang]} test
      mkdir train/audio
      mkdir test/audio
      down_sample.sh "$DIR/$lang/train/Audios/*" $DIR/$lang/train/'audio'/
      down_sample.sh "$DIR/$lang/test/Audios/*" $DIR/$lang/test/'audio'/
      rm -r train/Audios
      rm -r test/Audios
      touch ${DIR}/${lang}.done
    else
          echo "$lang data already exists. Skip prep."
    fi
done
```
<h4>Kaldi style data preparation</h4>
Now that the data is ready to be used, we are in the data preparation state where certain files are to be generated. The generated files are to be present as required in kaldi. All the steps are available <a href='https://kaldi-asr.org/doc/data_prep.html'>here</a>, at kaldi website. I will go through the specifics regarding subtask1 preparation.

- We need a ```text``` file which contains utterance id along with the corresponding text. If speaker information is present, it is should be prefix of utterance id    
CODE

- Next, we have ```wav.scp``` file. For the sake of simpkicity, we will have utterence id along with path to corresponding ```.wav``` file.
CODE

- We need a mapping from utterance id to speaker id, this is present in ```utt2spk```.
If no speaker information is present, speaker id can be same as utterence id.
CODE

- a ```segments``` file can be present which will contain begin and end time stamps for the audio.
CODE

Now, there are some files which will be generated automatically, there are as follows
- ```spk2utt``` is created from ```utt2spk```.
CODE

- The mapping between utterence id and computed features are present in ```feats.scp```. This is created while executing ```run.sh```.

To ensure that data is extrcated and mapped properly, following lines are present in ```run.sh```.
```
utils/validate_data_dir.sh data/train
utils/fix_data_dir.sh data/train
...
```
<h4>End-to-end integration for subtask1</h4>
With this, data preparation is complete. How do the different files interact with each other?

We store ```download_data.sh``` & ```prepare_data.sh``` in ```local/```.
Then it is handled in initial stages in ```run.sh```. We first download the data with ```download_data.sh``` in stage -1.
CODE

We will then prepare the kaldi data in stage 0.
CODE

After this, you can proceed with other stages starting with feature generation. We use fbank with pitch as audio features. After this is computed, it need not be visited again to change the model or fine tune results. So we can start from stage 2.

Experiment reuslts & logs are stored in `exp/` folder. Model configuration file can be added in ```conf/```, new experiment directories will be created inside `exp/` with the config file name present.


#show_results
<h4>ESPnet recipe subtask2</h4>
Those were the specifications for subtask1. What do we need to change for subtask2?

We first download the data, only the link has to be changed. Subtask2 data is already avaiable in kaldi style. So we do not need to generate them. The only change necessary is to add the current .wav path to ```wav.scp```. This is performed by the following code.
CODE

Once this is performed, the data is ready to be used by feature generation. The same steps are followed as before.

<h4>ESPnet2 recipes</h4>
Now that our espnet codes are ready to train ASR models, how can we transition to ESPnet2? It involved minimal steps. Here are the main design differences,

- Codes are present in ```egs2/```
- ```run.sh``` is reduced to passing arguements, no stages present.
- ```egs2/TEMPLATE/asr/``` contains the common code to be used by all recipies.
- All steps involving data download and preparation from ```run.sh``` is now present in ```local/data.sh```.

<h4>ESPnet2 recipe subtask1</h4>
Here, data.sh becomes,
CODE

And, ```run.sh``` is as follows

In this case, once the features are computed, ```SKIP____``` flag can be turned to false to deiable recomputing in further runs. config files are present in a slightly different structure & certain parameters may need to be renamed

Eg: ```grad-clip -> grad_clip```
<h4>ESPnet2 recipe subtask2</h4>
Here, data.sh becomes,
CODE

And, ```run.sh``` is as follows

Additional info

Somthing to conclude
