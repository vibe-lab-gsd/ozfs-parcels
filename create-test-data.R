library(here)
library(tidyverse)
library(sf)
library(tigris)

ellis_parcels <- here("ellis_co_shp") |>
  st_read()

ennis <- places(state = "TX") |>
  filter(NAME == "Ennis") |>
  st_transform(st_crs(ellis_parcels))

ennis_parcels <- ellis_parcels |>
  st_filter(ennis) 

ennis_parcels$unique_id <- seq(1, nrow(ennis_parcels), by=1)

random_select <- sample(ennis_parcels$unique_id, size = 50)

fifty_parcels <- ennis_parcels |>
  filter(unique_id %in% random_select)

roads <- roads(state = "TX", county = "Ellis")  |>
  st_transform(st_crs(ellis_parcels)) |>
  st_filter(ennis)

st_write(roads, 
         here("test-data", "roads.geojson"))

st_write(fifty_parcels,
         here("test-data", "parcels.geojson"))
