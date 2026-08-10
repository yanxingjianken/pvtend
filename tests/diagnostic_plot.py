import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_diagnostic():
    base_path = "/net/flood/data2/users/x_yan/pvtend/examples/era5_jan2025"
    z_file = os.path.join(base_path, "era5_z_2025_01.nc")
    pv_file = os.path.join(base_path, "era5_pv_2025_01.nc")
    output_path = "/net/flood/data2/users/x_yan/tmp/cali_block_z_pv_contour_diagnostic.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    time_idx = 252
    center_lat, center_lon = 40, -120
    lat_range = 21
    lon_range = 36
    g = 9.80665

    try:
        ds_z = xr.open_dataset(z_file).isel(time=time_idx)
        ds_pv = xr.open_dataset(pv_file).isel(time=time_idx)
        
        # Handle longitude wrapping if necessary (ERA5 is usually 0-360)
        if ds_z.longitude.max() > 180 and center_lon < 0:
            ds_z = ds_z.assign_coords(longitude=(((ds_z.longitude + 180) % 360) - 180)).sortby('longitude')
            ds_pv = ds_pv.assign_coords(longitude=(((ds_pv.longitude + 180) % 360) - 180)).sortby('longitude')

        def extract_patch(ds):
            return ds.sel(
                latitude=slice(center_lat + lat_range, center_lat - lat_range),
                longitude=slice(center_lon - lon_range, center_lon + lon_range)
            )

        patch_z = extract_patch(ds_z)
        patch_pv = extract_patch(ds_pv)

        fig, axs = plt.subplots(2, 2, figsize=(16, 12), facecolor='black')
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

        levels_to_plot = [500, 300]
        variables = [('Z', patch_z, 'z'), ('PV', patch_pv, 'pv')]
        
        for i, level in enumerate(levels_to_plot):
            # Z panels
            ax_z = axs[0, i]
            data_z = patch_z['z'].sel(level=level) / g
            im_z = ax_z.contourf(patch_z.longitude, patch_z.latitude, data_z, cmap='magma', levels=20)
            cs_z = ax_z.contour(patch_z.longitude, patch_z.latitude, data_z, colors='white', linewidths=0.5, levels=15)
            ax_z.clabel(cs_z, inline=True, fontsize=8, fmt='%1.0f')
            ax_z.set_title(f"Z{level} (m)", color='white')
            ax_z.set_facecolor('black')
            print(f"Z{level} range: {data_z.min().values:.2f} to {data_z.max().values:.2f}")

            # PV panels
            ax_pv = axs[1, i]
            data_pv = patch_pv['pv'].sel(level=level)
            if data_pv.max() < 1e-3: # Check if in SI units
                data_pv = data_pv / 1e-6
            im_pv = ax_pv.contourf(patch_pv.longitude, patch_pv.latitude, data_pv, cmap='viridis', levels=20)
            cs_pv = ax_pv.contour(patch_pv.longitude, patch_pv.latitude, data_pv, colors='white', linewidths=0.5, levels=15)
            ax_pv.clabel(cs_pv, inline=True, fontsize=8, fmt='%1.2f')
            ax_pv.set_title(f"PV{level} (PVU)", color='white')
            ax_pv.set_facecolor('black')
            print(f"PV{level} range: {data_pv.min().values:.2f} to {data_pv.max().values:.2f}")

        for ax in axs.flat:
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')

        plt.savefig(output_path, facecolor='black', bbox_inches='tight')
        print(f"Saved diagnostic plot to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_diagnostic()
