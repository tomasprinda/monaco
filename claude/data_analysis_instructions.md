# Data Analysis instructions

This document covers the instruction for running the data analysis. 

## Input
Input of the data analysis is a brief description of what analysis to do


## Output: 
Set of files in the output directory. 
- The main output of the analysis is one image that can be 
shared on the social media. Ideally there should be one major takeaway. 
There should be a title, subtitle, one chart, possibly additional information,
and sources at the bottom.
- There should be html file that that describes the analysis in details. There should be
  - the main image 
  - possible alternatives. 
  - Data sources that has been used (url, sample data, basic stats about the data)
  - Assumptions
  - Calculations 
  - Step by step what was done
  - Why it should be interesting for the target customer? 
  - Your thought about the analysis.
    
## Target customer:

The final post should be interesting for people who rent properties.
They probably own 1-10 properties that they rent (mainly flats). Or perhaps
people that are thinking about investing in properties.
They have some problems, they want to know how to invest next. 
The posts can even be about something that makes them surprised, curious about something,
learn something they wouldnt expect

## Steps to perform the analysis:

### 1. Identify data sources

The data sources should be reliable. Ideally officially published by reliable institutions. 
Some examples are:
- Český statistický úřad

There can be multiple sources.

### 2. Download the data

Data should be downloaded to data/ subdirectory. It should be organized as
`data/raw/<analysis_name>/<timestamp>/<data_source>/`.
The data should be downloaded in the reasonably raw format.

### 3. Process data

Data should be cleaned, only the important parts should be kept. Data may be 
reorganized to different format that is more suitable for the downstream analysis.

### 4. Analyze data

Create a few candidate plots that can become the main output. 
Generate also some supporting statistics and information.

### 5. Asses and iterate

Analyze the results. It it trustworthy? Is it interesting for the target customer?
Does he learn something new? Does it solve some of his problem? Does it
help him to make a decision?

Based on the findings feel free to refine and change the analysis. 


### 6. Generate outputs

