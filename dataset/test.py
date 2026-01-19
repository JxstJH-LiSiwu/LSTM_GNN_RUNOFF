import geopandas as gpd
gdf = gpd.read_file("A_basins_total_upstrm/3_shapefiles/river_network.shp")
print(gdf.columns)
print(gdf.head(3))
