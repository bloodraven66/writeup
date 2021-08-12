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

#compute kaldi data

#using it in ./run.sh

#using ./run.sh while fine tuning (what stages to skip, etc)

#show_results

#subtask2

#what changes here and why?

#different changes from egs to eg2
#show reuse older codes for this

#subtask2
