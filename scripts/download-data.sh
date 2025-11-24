#!/bin/bash

# get data from kaggle, unzip and finally delete de zip file
kaggle datasets download olistbr/brazilian-ecommerce -p ./data/raw --unzip
