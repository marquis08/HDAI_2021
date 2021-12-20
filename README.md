# 대회 정보
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://www.kaggle.com/hominlee"><img src="https://avatars.githubusercontent.com/u/33175883?v=4?s=100" width="100px;" alt=""/><br /><sub><b>DShomin</b></sub></a><br /><a href="https://github.com/marquis08/HDAI_2021/commits?author=DShomin" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/hyeonho1028"><img src="https://avatars.githubusercontent.com/u/40379485?v=4?s=100" width="100px;" alt=""/><br /><sub><b>hyeonho lee</b></sub></a><br /><a href="https://github.com/marquis08/HDAI_2021/commits?author=hyeonho1028" title="Code">💻</a></td>
    <td align="center"><a href="https://www.kaggle.com/marquis08"><img src="https://avatars.githubusercontent.com/u/27425140?v=4?s=100" width="100px;" alt=""/><br /><sub><b>YILGUK SEO</b></sub></a><br /><a href="https://github.com/marquis08/HDAI_2021/commits?author=marquis08" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!