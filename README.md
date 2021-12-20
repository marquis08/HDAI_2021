# 대회 정보
- <http://hdaidatathon.com/>
- https://github.com/DatathonInfo/H.D.A.I.2021

# Directory Substructure

```
├── data
│   ├── train
│   └── validation
├── src
├── install.sh
└── requirements.txt
```


# Preprocessing(dehaze)
```sh
python create_df.py -C config
python dehaze_preprocess.py
```

# Train
```sh
python main.py -C config
```

# Inference (ensemble)
```sh
python ensemble.py -C config
```
