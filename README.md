# NOAA WEATHER FORECAST PROJECT
**Author: Ebuwa Evbuoma-Fike**

**Last Edited: 07/15/2022**

**Purpose:**
Predict local weather (temperature parameters) using ridge regression modeling. This project was inspired by Dataquest.io's repository.

**Data Source**

NOAA Public Data Sets, Daily Weather
https://www.ncdc.noaa.gov/cdo-web/search

**Variables**

- STATION_NAME (max 50 characters) is the name of the station (usually city/airport name). Optional
output field.
- DATE is the year of the record (4 digits) followed by month (2 digits) and day (2 digits).

The five core weather values:
- PRCP = Precipitation (mm or inches as per user preference, inches to hundredths on Daily Form pdf file)
- SNOW = Snowfall (mm or inches as per user preference, inches to tenths on Daily Form pdf file)
- SNWD = Snow depth (mm or inches as per user preference, inches on Daily Form pdf file)
- TMAX = Maximum temperature (Fahrenheit or Celsius as per user preference, Fahrenheit to tenths on Daily Form pdf file
- TMIN = Minimum temperature (Fahrenheit or Celsius as per user preference, Fahrenheit to tenths on Daily Form pdf file

All other weather values:

- ACMC = Average cloudiness midnight to midnight from 30-second ceilometer data (percent)
- ACMH = Average cloudiness midnight to midnight from manual observations (percent)
- ACSH = Average cloudiness sunrise to sunset from manual observations (percent)
- AWND = Average daily wind speed (meters per second or miles per hour as per user preference)
- FMTM = Time of fastest mile or fastest 1-minute wind (hours and minutes, i.e., HHMM)
- FRGT = Top of frozen ground layer (cm or inches as per user preference)
- PGTM = Peak gust time (hours and minutes, i.e., HHMM)
- PSUN = Daily percent of possible sunshine (percent)
- TSUN = Daily total sunshine (minutes)
- WDF1 = Direction of fastest 1-minute wind (degrees)
- WDF2 = Direction of fastest 2-minute wind (degrees)
- WDF5 = Direction of fastest 5-second wind (degrees)
- WDFG = Direction of peak wind gust (degrees)
- WDFI = Direction of highest instantaneous wind (degrees)
- WDFM = Fastest mile wind direction (degrees)
- WESD = Water equivalent of snow on the ground (inches or mm as per user preference)
- WSF1 = Fastest 1-minute wind speed (miles per hour or meters per second as per user preference)
- WSF2 = Fastest 2-minute wind speed (miles per hour or meters per second as per user preference)
- WSF5 = Fastest 5-second wind speed (miles per hour or meters per second as per user preference)
- WSFG = Peak guest wind speed (miles per hour or meters per second as per user preference)
- WSFM = Fastest mile wind speed (miles per hour or meters per second as per user preference)
- WT** = Weather Type where ** has one of the following values: 
-  01 = Fog, ice fog, or freezing fog (may include heavy fog)
-  02 = Heavy fog or heaving freezing fog (not always distinguished from fog)
-  03 = Thunder
-  04 = Ice pellets, sleet, snow pellets, or small hail
-  05 = Hail (may include small hail)
-  06 = Glaze or rime
-  07 = Dust, volcanic ash, blowing dust, blowing sand, or blowing obstruction
-  08 = Smoke or haze
-  09 = Blowing or drifting snow
-  10 = Tornado, waterspout, or funnel cloud
-  11 = High or damaging winds
-  12 = Blowing spray
-  13 = Mist
-  14 = Drizzle
-  15 = Freezing drizzle
-  16 = Rain (may include freezing rain, drizzle, and freezing drizzle)
-  17 = Freezing rain
-  18 = Snow, snow pellets, snow grains, or ice crystals
-  19 = Unknown source of precipitation
-  21 = Ground fog
-  22 = Ice fog or freezing fog

- WVxx = Weather in the Vicinity where “xx” has one of the following values
- 01 = Fog, ice fog or freezing fog (may include heavy fog)
- 03 = Thunder

Detailed documentation: https://www.ncei.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf
