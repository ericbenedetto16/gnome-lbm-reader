from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

class GNOMEVelocityReader:
    def __init__(self, file_name):
        self.file_name = file_name
        self.reader = Dataset(file_name, 'r')
        self.times = self.reader.variables['time'][:]
        self.lats = self.reader.variables['lat'][:]
        self.lngs = self.reader.variables['lon'][:]
        self.water_u = self.reader.variables['water_u'][:]
        self.water_v = self.reader.variables['water_v'][:]

    def get_times(self):
        return self.times

    # Latitudes and Longitudes
    def get_locations(self):
        return (self.lats, self.lngs)

    # Eastward and Northward Water Velocity, Respectively
    def get_velocities(self):
        return (self.water_u, self.water_v)

    def close(self):
        self.reader.close()

class GNOMEParticleReader:
    def __init__(self, file_name, grid_size=200):
        self.grid_size = grid_size
        self.file_name = file_name
        self.reader = Dataset(file_name, 'r')
        self.times = self.reader.variables['time'][:]
        self.particle_counts = self.reader.variables['particle_count'][:]
        self.particle_masses = self.reader.variables['mass'][:]
        self.particle_depths = self.reader.variables['depth'][:]
        self.particle_ages = self.reader.variables['age'][:]
        self.particle_statuses = self.reader.variables['status_codes'][:]
        self.particle_ids = self.reader.variables['id'][:]
        self.lats = self.reader.variables['latitude'][:]
        self.lngs = self.reader.variables['longitude'][:]

        # Reshape Lat/Lng to Be Store By Timestep
        self.np_lats = np.reshape(self.lats, (self.particle_counts[0], self.times.size))
        self.np_lngs = np.reshape(self.lngs, (self.particle_counts[0], self.times.size))

        self.min_lat = None
        self.max_lat = None
        self.min_lng = None
        self.max_lng = None

        # Calculate Min and Max Lat
        for i in range(0, self.get_times().size):
            for (lng, lat) in zip(self.np_lngs[:][i], self.np_lats[:][i]):
                if self.min_lat is None or lat < self.min_lat:
                    self.min_lat = lat
                if self.max_lat is None or lat > self.max_lat:
                    self.max_lat = lat

                if self.min_lng is None or lng < self.min_lng:
                    self.min_lng = lng
                if self.max_lng is None or lng > self.max_lng:
                    self.max_lng = lng

        # Store the Bounding Box from the GNOME Simulation
        self.boundingBox = [(self.min_lat, self.min_lng), (self.max_lat, self.max_lng)]

        # Determine The Delta for Each Grid Cell Based on Range of Lat/Lng Respectively
        self.lat_ticks = (self.max_lat - self.min_lat) / self.grid_size
        self.lng_ticks = (self.max_lng - self.min_lng) / self.grid_size

        # Generate Bins for NxN Grid
        self.bins = np.zeros((self.grid_size, self.grid_size))

    def get_lat_bin(self, lat):
        if lat == self.max_lat:
            return self.grid_size - 1
        return int(np.floor((lat - self.min_lat) / self.lat_ticks))
    
    def get_lng_bin(self, lng):
        if lng == self.max_lng:
            return self.grid_size - 1
        return int(np.floor((lng - self.min_lng) / self.lng_ticks))
         
    def get_times(self):
        return self.times

    def get_locations(self):
        return (self.lats, self.lngs)

    def get_particle_counts(self):
        return self.particle_counts

    def get_particle_masses(self):
        return self.particle_masses

    def get_particle_depth(self):
        return self.particle_depths

    def get_particle_ages(self):
        return self.particle_ages

    def get_particle_statuses(self):
        return self.particle_statuses

    def get_particle_ids(self):
        return self.particle_ids

    # Generate and  Return the NxN Matrix with Particle Counts at Timestep i
    def generate_lbm_particle_count_matrix(self, i):
        local_bins = np.array(self.bins, dtype=int)
        for (lng, lat) in zip(self.np_lngs[:][i], self.np_lats[:][i]):
            lat_bin = self.get_lat_bin(lat)
            lng_bin = self.get_lng_bin(lng)

            local_bins[lat_bin][lng_bin] = local_bins[lat_bin][lng_bin] + 1
        
        return local_bins

    # 3D Plot the Particle Counts in an NxN Grid as Would be in LBM
    def show_particle_frequencies(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        for i in range(0, self.times.size):
            plt.cla()
            plt.title(f"Particle Distribution at t={i}")

            lbm_particle_count = self.generate_lbm_particle_count_matrix(i)
            
            x = np.arange(self.grid_size)
            y = np.arange(self.grid_size)
            X, Y = np.meshgrid(x, y)

            ax.plot_surface(X, Y, lbm_particle_count, cmap=plt.get_cmap('gist_earth'))
            plt.pause(0.5)

        plt.show()

    # 2D Plot Particles at their lat/lng
    def show_particle_trace(self):
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        for i in range(0, self.times.size):
            plt.cla()
            plt.scatter(self.np_lngs[:][i],self.np_lats[:][i], color='b', s=100, marker='.')

            plt.title(f"Particle Trace at t={i}")
            plt.pause(0.50)
            
        plt.show()

    def close(self):
        self.reader.close()

# Driver Code
velocity_reader = GNOMEVelocityReader("Gnome26x26V5.nc")
particle_reader = GNOMEParticleReader("input_file.nc")

# Generate LBM Data for Time t
t = 200
lbm_at_200 = particle_reader.generate_lbm_particle_count_matrix(t)

# Export LBM Data to File
outfile = open(f"lbm_at_{t}.txt", 'w')
np.savetxt(outfile, lbm_at_200, fmt='%1d')
outfile.close()

# Example of How to Read Dumped LBM Data in Another Program
lbm_read = np.loadtxt(f"lbm_at_{t}.txt")

particle_reader.show_particle_frequencies()
# particle_reader.show_particle_trace()

velocity_reader.close()
particle_reader.close()