##Authors: Sayan Kr. Swar, Tushar Mittal, Tolulope Olugboji

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
import re
from PIL import Image
import numpy as np

from jax import jit
from jax import numpy as jnp
from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import *
from jwave.utils import show_field, show_positive_field
from jwave.signal_processing import analytic_signal, apply_ramp, gaussian_window, smooth

import networkx as nx
import structify_net.zoo as zoo

import gstools as gst
from gstools import transform as tf
from gstools.random import MasterRNG


class Simulator:
    def __init__(self, domain=None, density_rho=1.0, sound_speed=1500.0, pml_size=5, num_air_grid=10, base_density=3000, defect_density=1000, air_speed_of_sound=343.0):
        # #### Model Inputs
        # k: This parameter is specific to the Watts-Strogatz model and represents the number of nearest neighbors each node is initially connected to in a ring structure. While it's defined here, it's important to note the comment stating it's used only for the 'small_world' graph type, which is not the type being generated in this specific block.
        # p_vals: This variable represents the probability, although its exact interpretation depends on the graph generation function used. In the context of structify_net graphs, it seems to relate to the density of the graph (proportion of possible edges that exist).
        # size: This sets the number of nodes in the graph, which is also interpreted as the dimension of the resulting image (size x size pixels).
        # epsilon: This parameter is used by the structify_net graph generation functions and controls the "randomness" or deviation from the ideal structure of the specified graph type.
        # graph_type: This string specifies the type of graph to generate. In this case, it's set to 'overlapping_communities', which indicates that a graph with overlapping community structure will be created using the structify_net library.
        # Generate a random graph using the Watts-Strogatz model (Small-world graph)

        if domain is None:
            self.domain = Domain((128*2, 128*2), (1, 1))
        else :
            self.domain = domain
        self.num_air_grid = num_air_grid
        self.base_density = base_density # m/s
        self.defect_density = defect_density # m/s
        self.air_speed_of_sound = air_speed_of_sound # m/s

        self.k = 20  # Each node is joined to k nearest neighbors in a ring topology (used only for small_world)
        self.m = 10
        self.p_vals = 0.1 # The probability of rewiring each edge
        self.epsilon = .5
        self.size = self.domain.N[0]-2*self.num_air_grid ## Number of pixels - images is size x size
        #### Note that structify sometimes fails for values of the larger values of the size
        # (or works best for some round numbers) -- so, for larger images, one can rescale or stack.
        self.graph_type = 'fractal_root'

        self.density_rho = density_rho
        self.sound_speed = sound_speed

        # We have to update the attenutation function based on user input
        attentuation = jnp.ones(self.domain.N)
        self.attentuation = FourierSeries(jnp.expand_dims(attentuation, -1), self.domain)
        # attenuation = jnp.expand_dims(attenuation.at[64:110, 125:220].set(100), -1)

        self.pml_size = pml_size
        self.time_axis = None
        self.sensors = None
        self.sources = None
        self.use_sensors = False
        self.use_sources_Time = False
        self.use_sources_p0 = False
        self.path = ''
        self.set_sound_flag = True 
        self.set_density_flag = False
        self.G_grid_array = None
        self.num_sensors = 48

        self.cov_model_name = 'Gaussian'
        self.cov_model_dim = 2
        self.cov_model_var = 1
        self.cov_model_len_scale = 15
        self.cov_model_angles = np.pi
        self.cov_model_transform_field = False


    def run_basic_setup(self, circle=False, use_p0=True, field_type='Graph', **kwargs):
      assert field_type in ['Graph','Spatial'], 'not a valid field type'
      plot_data = kwargs.get('plot_data', False) 
      plot_variogram = kwargs.get('plot_variogram', False) 
      plot_only_graph = kwargs.get('plot_only_graph', False) 
      save_summary_plot = kwargs.get('save_summary_plot', False)
      add_sensor_at_source = kwargs.get('add_sensor_at_source', False)
      
      self.field_type = field_type
      if self.field_type=='Graph':
        self.generate_graph(plot_me=plot_only_graph)
      else:
        transform_mode = kwargs.get('transform_mode', ['Binary'])
        seed_val = kwargs.get('seed_val', MasterRNG(None)())  
        _,_,_ = self.generate_spatial_corr_field(cov_model=self.cov_model_name,dim=self.cov_model_dim,
                          var=self.cov_model_var,len_scale=self.cov_model_len_scale,angles=self.cov_model_angles,
                          transform_field=self.cov_model_transform_field,
                          transform_mode=transform_mode,seed_val=seed_val,plot_field=plot_variogram)

      self.set_density()
      self.set_time_arr()

      if use_p0:
        self.source_type = "pressure"
        pressure_source_raddi = kwargs.get('pressure_source_raddi', 6) 
        pressurce_source_loc = kwargs.get('pressurce_source_loc', (62,62)) 
        self.generate_sources_p0(radius=pressure_source_raddi,center_x=pressurce_source_loc[0],center_y=pressurce_source_loc[1])
      else:
        self.source_type = "gaussian pulse"
        source_loc = kwargs.get('pulse_source_loc', ((62,62),(62,62))); assert len(source_loc)==2,'Please pass two different pulse sources in ((x1,y1),(x2,y2)) format. Both location can be of same value.'
        amp = kwargs.get('amp', 1);  freq = kwargs.get('freq', 50);  
        pulse_sigma = kwargs.get('pulse_sigma', 4e-2);  pulse_1_m = kwargs.get('pulse_1_m', .08);  pulse_2_m = kwargs.get('pulse_2_m', .52);  
        self.generate_sources_time(source_loc, amp, freq, pulse_sigma, pulse_1_m, pulse_2_m)

      if circle:
        self.sensor_allignment = "circular"
        self.circle_dim = kwargs.get('circle_dim', (125,125)) 
        self.generate_sensors_circle(num_sensors = self.num_sensors, circle_x=self.circle_dim[0],circle_y=self.circle_dim[1],add_sensor_at_source=add_sensor_at_source)
      else:
        self.sensor_allignment = "linear"
        self.sides = kwargs.get('sensor_locations', ['top', 'right', 'bottom', 'left']) 
        self.generate_sensors_edges(sides=self.sides,offset=self.num_air_grid+5,spacing_samples=6,add_sensor_at_source=add_sensor_at_source)
      
      if plot_data:
        self.plot_summary(save_me=save_summary_plot)
      


    def generate_graph(self, plot_me):
      gpx, _ =  self._generate_graph(self.graph_type,self.size,epsilon=self.epsilon,p=self.p_vals,k=self.k,m=self.m)
      ## Get the adjacency matrix of the graph
      if self.G_grid_array is None:
        self.G_grid_array = nx.to_numpy_array(gpx)
      
      if plot_me:
        plt.figure(figsize=(10,10))
        plt.imshow(1-self.G_grid_array,cmap='grey')
        plt.colorbar()
        plt.show()
    
    def _generate_graph(self, graph_type, size, scores=False, verbose=False, **kwargs):
      """
      Generate a graph of the specified type.

      Parameters:
      graph_type (str): The type of graph to generate (e.g. 'random', 'scale_free', 'small_world').
      size (int): The number of nodes in the graph.
      **kwargs: Additional keyword arguments for the graph generation function.

      Returns:
      G (networkx.Graph): The generated graph.
      """
      p_vals = kwargs.get('p', 0.4)
      structufy_keys = ['ER', 'spatial', 'spatialWS', 'blocks_assortative', 'overlapping_communities', 'nestedness', 'maximal_stars', 'core_distance',
                        'fractal_leaves', 'fractal_root', 'fractal_hierarchy', 'fractal_star', 'perlin_noise', 'disconnected_cliques']

      if graph_type == 'erdos_renyi_graph':
          return nx.erdos_renyi_graph(int(size),p_vals),0 #p=1 -> all-to-all connectivity
      elif graph_type == 'random':
          return nx.gnm_random_graph(int(size), kwargs.get('num_edges', 500)),0
      elif graph_type == 'scale_free':
          return nx.barabasi_albert_graph(int(size), kwargs.get('m', 2)),0
      elif graph_type == 'small_world':
          return nx.watts_strogatz_graph(int(size), kwargs.get('k', 4), p_vals),0
      elif graph_type in structufy_keys:
          try:
              if verbose:
                  print(f'Size of graph : {size}, non-zero values : {int(p_vals*size**2.)}') 
              list_zoo = zoo.get_all_rank_models(n=size,m=int(p_vals*size**2.)) 
              using_size_diff = False
          except:
              size_use = np.max([50,size])
              list_zoo = zoo.get_all_rank_models(n=size_use,m=int(p_vals*size_use**2.))
              using_size_diff = True
              print('HERE in Zoo',size,int(p_vals*size**2.))
          
          #print(list_zoo.keys())
          rank_model = list_zoo[graph_type]
          epsilon = kwargs.get('epsilon', 0.1)
          if scores :
              df_scores_graph = rank_model.scores(m=int(p_vals*size**2.),epsilons=epsilon,runs=10)
          else :
            df_scores_graph = None
          if using_size_diff :
              gpx =  rank_model.generate_graph(epsilon=epsilon,density=p_vals)
              # Get the adjacency matrix of the graph
              A = nx.to_numpy_array(gpx)
              n_avg = size
              # Calculate the block size for averaging
              block_size = int(np.ceil(A.shape[0] / n_avg))
              # Calculate the padded size
              padded_size = block_size * n_avg
              # Calculate the number of repetitions needed to reach the padded size
              repetitions = int(np.ceil(padded_size / A.shape[0]))
              # Repeat the original matrix to create a larger matrix
              A_repeated = np.tile(A, (repetitions, repetitions))
              # Trim the repeated matrix to the padded size
              A_padded = A_repeated[:padded_size, :padded_size]
              # Average the adjacency matrix locally to make it a n_avg x n_avg matrix
              A_avg = np.mean(np.mean(A_padded.reshape(n_avg, block_size, n_avg, block_size), axis=1), axis=2)
              return nx.from_numpy_array(A_avg),df_scores_graph
          else :
              return rank_model.generate_graph(epsilon=epsilon,density=p_vals),df_scores_graph
      else:
          raise ValueError(f"Invalid graph type: {graph_type}")


    def generate_spatial_corr_field(self,cov_model='Gaussian',dim=2,var=1,len_scale=5,angles=np.pi,transform_field=False,**kwargs):
      assert cov_model in ['Gaussian','Exponential','Matern','Spherical','Circular','Linear','Stable'], 'Undefined Covariance Model'
      seed_val = kwargs.get('seed_val', MasterRNG(None)())  
      plot_field = kwargs.get('plot_field', False)  

      x = y = range(self.size)
      self.graph_type = cov_model

      model_map = {
        'Gaussian': lambda dim, var, len_scale, angles: gst.Gaussian(dim=dim, var=var, len_scale=len_scale, angles=angles),
        'Exponential': lambda dim, var, len_scale, angles: gst.Exponential(dim=dim, var=var, len_scale=len_scale, angles=angles, anis=0.5),
        'Matern': lambda dim, var, len_scale, angles: gst.Matern(dim=dim, var=var, len_scale=len_scale, angles=angles),
        'Spherical': lambda dim, var, len_scale, angles: gst.Spherical(dim=dim, var=var, len_scale=len_scale, angles=angles),
        'Circular': lambda dim, var, len_scale, angles: gst.Circular(dim=dim, var=var, len_scale=len_scale, angles=angles),
        'Linear': lambda dim, var, len_scale, angles: gst.Linear(dim=dim, var=var, len_scale=len_scale, angles=angles),
        'Stable': lambda dim, var, len_scale, angles: gst.Stable(dim=dim, var=var, len_scale=len_scale, angles=angles)
      }

      model_to_call = model_map.get(cov_model, lambda dim, var, len_scale, angles: gst.Gaussian(dim=dim, var=var, len_scale=len_scale, angles=angles))
      model = model_to_call(dim, var, len_scale, angles)
      srf = gst.SRF(model, seed=seed_val)
      if not transform_field:
        field_arr = srf.structured([x, y])
      else:
        transform_mode = kwargs.get('transform_mode', ['Binary'])
        transfrom_map = {
        'Binary': lambda model: tf.binary(model),
        'ZinnHarvey': lambda model: tf.zinnharvey(model),
        'LogNormal': lambda model: tf.normal_to_lognormal(model),
        'ForceMoment': lambda model: tf.normal_force_moments(model)
        }
        srf.structured([x, y])
        for i in range(0,len(transform_mode)):
          transformed_call = transfrom_map.get(transform_mode[i], lambda model: tf.binary(model))
          field_arr = transformed_call(srf)

      min_val = np.min(field_arr)
      max_val = np.max(field_arr)
      #normalized_field_arr = 2 * ((field_arr - min_val) / (max_val - min_val)) - 1
      range_val = max_val - min_val; denominator = range_val if range_val != 0 else 1e-6;
      normalized_field_arr = (field_arr - min_val) / denominator
      
      if self.G_grid_array is None:
        self.G_grid_array = normalized_field_arr.T

      if plot_field:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.contourf(x, y, normalized_field_arr.T, levels=256); plt.title(f'{cov_model} Field')
        plt.colorbar()
        plt.title(f'spatially correlated field (normalized)')
        plt.subplot(1,2,2)
        plt.plot(model.variogram(np.linspace(0,100,1000)),label=f'{cov_model} Variogram')
        plt.legend()

      return normalized_field_arr.T, field_arr.T, (x, y)


    def set_density(self):
      density_field = self._set_density_field(self.domain,self.num_air_grid,self.base_density,self.defect_density,self.air_speed_of_sound,self.G_grid_array)
      if self.set_sound_flag:
        self.sound_speed = density_field
      else:
        self.density_rho = density_field
      #print(f"Set density data for Sound : {self.set_sound_flag}, Density : {self.set_density_flag}")

    def _set_density_field(self, domain,num_air_grid,base_density,defect_density,air_speed_of_sound,G_grid_array,plot=False):
      ## Setting the density field from the graph data and other parameters
      density = jnp.zeros(domain.N)
      if self.field_type == 'Graph':
        G_grid_jax = jnp.array((1-G_grid_array)*base_density)
        density += density.at[num_air_grid:-num_air_grid, num_air_grid:-num_air_grid].set(G_grid_jax)
        G_grid_jax = jnp.array(G_grid_array*defect_density)
        density += density.at[num_air_grid:-num_air_grid, num_air_grid:-num_air_grid].set(G_grid_jax)
      else:
        G_grid_jax = jnp.array((1 - G_grid_array) * base_density + G_grid_array * defect_density) 
        #jnp.array(G_grid_array*defect_density) + base_density
        density += density.at[num_air_grid:-num_air_grid, num_air_grid:-num_air_grid].set(G_grid_jax)

      density = density.at[0:num_air_grid, :].set(air_speed_of_sound)
      density = density.at[:, 0:num_air_grid].set(air_speed_of_sound)
      density = density.at[-num_air_grid:, :].set(air_speed_of_sound)
      density = density.at[:, -num_air_grid:].set(air_speed_of_sound)
      density_field = FourierSeries(np.expand_dims(density, -1), domain)

      if plot:
        show_positive_field(density_field)
        _ = plt.title("Density")
      return density_field


    def set_time_arr(self):
      medium = Medium(domain=self.domain, sound_speed=self.sound_speed, density=self.density_rho, pml_size=self.pml_size, attenuation=self.attentuation)
      self.time_axis = TimeAxis.from_medium(medium, cfl=0.3)
      self.time_axis_arr = self.time_axis.to_array()
      self.medium = Medium
      #print(f'time_axis_arr Shape : {self.time_axis_arr.shape}')
    

    def generate_sources_p0(self,radius=6,center_x=62,center_y=62):
      ### Pressure source properties
      self.radius = radius
      self.center_x = center_x
      self.center_y = center_y

      p0 = self._points_on_circle_press(self.radius,self.center_x,self.center_y,self.domain.N)
      self.p0 = p0
      self.use_sources_p0 = True
    
    def _points_on_circle_press(self,radius,center_x,center_y,N,amplt=1.,plot=False):
      # Defining the initial pressure
      p0 = circ_mask(N, radius, (center_x,center_y))
      p0 = amplt * jnp.expand_dims(p0, -1)
      p0 = FourierSeries(p0, self.domain)
      if plot:
        show_field(p0)
        plt.title("Initial pressure")
      return p0


    def generate_sources_time(self, source_loc=((62,62),(62,62)), amp=10, freq=50, pulse_sigma=4e-2, pulse_1_m = .08, pulse_2_m=.52):
        self.radius = 1
        self.center_x = source_loc[0][0]
        self.center_y = source_loc[0][1]
        self.pulse_source_loc = source_loc

        t = np.arange(0, self.time_axis.t_end, self.time_axis.dt)
        s = np.sin(2 * np.pi * freq * t)
        s1 = gaussian_window(s, t, pulse_1_m, pulse_sigma)
        s2 = gaussian_window(s, t, pulse_2_m, pulse_sigma)
        self.sources =  Sources(
            positions=self.pulse_source_loc,
            signals=jnp.stack([amp*s1, amp*s2]),
            dt=self.time_axis.dt,
            domain=self.domain,
        )
        self.use_sources_Time = True
        #plt.plot(s1)
        #plt.plot(s2)


    def generate_sensors_circle(self, num_sensors, sensor_radius=100, circle_x=100, circle_y=100, add_sensor_at_source=False):
      ## Sensors
      self.num_sensors = num_sensors
      self.sensor_radius = sensor_radius
      self.circle_x = circle_x
      self.circle_y = circle_y

      x, y = points_on_circle(num_sensors, sensor_radius, (circle_x, circle_y))
      print(num_sensors)
      if add_sensor_at_source:
        # Add One Sensor at Source with 1 Offset
        x = (self.radius+self.center_x,) + x
        y = (self.radius+self.center_y,) + y

      sensors_positions = (x, y)
      self.sensors = Sensors(positions=sensors_positions)
      #print("Sensors parameters: ",Sensors.__annotations__)
      self.use_sensors=True
      self.x = x
      self.y = y
      self.number_sensors = len(self.x)


    def generate_sensors_edges(self, arr=None, sides=['top', 'right', 'bottom', 'left'], offset=10, spacing_samples=1, add_sensor_at_source=False):
      if arr is None:
        arr = np.zeros(self.domain.N)
      # Example usage:
      points = self._get_edge_points(arr, offset, sides, spacing_samples=spacing_samples)
      x, y = zip(*points)
      if add_sensor_at_source:
        # Add One Sensor at Source with Offset
        x = (self.radius+self.center_x,) + x
        y = (self.radius+self.center_y,) + y

      sensors_positions = (x, y)
      self.sensors = Sensors(positions=sensors_positions)
      #print("Sensors parameters: ",Sensors.__annotations__)
      self.use_sensors=True
      self.x = x
      self.y = y
      self.number_sensors = len(self.x)
    
    def _get_edge_points(self, arr, offset, sides,spacing_samples=1):
      """
      Get x-y points at the edges of a 2D array with a specified offset.

      Parameters:
      arr (2D array): Input array
      offset (int): Offset from the edge
      sides (list of str): Which sides to include ('top', 'bottom', 'left', 'right')

      Returns:
      points (list of tuples): x-y points at the edges
      """
      rows, cols = arr.shape
      points = []

      if 'top' in sides:
          points.extend([(i, offset) for i in range(offset, cols - offset,spacing_samples)])
      if 'bottom' in sides:
          points.extend([(i, rows - offset - 1) for i in range(offset, cols - offset,spacing_samples)])
      if 'left' in sides:
          points.extend([(offset, i) for i in range(offset, rows - offset,spacing_samples)])
      if 'right' in sides:
          points.extend([(cols - offset - 1, i) for i in range(offset, rows - offset,spacing_samples)])

      return points


    def plot_summary(self,save_me):
      if self.use_sources_p0 == False and self.use_sources_Time == False:
        return ValueError('No source field set - use generate_sources_p0/time()')

      #fig, ax = plt.subplots(1,2, figsize=(15,10), dpi=100)
      fig = plt.figure(figsize=(10, 6), dpi=100)

      gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

      if self.use_sources_p0:
          ax0 = fig.add_subplot(gs[0])
          im1 = ax0.imshow(self.p0.on_grid, cmap="RdBu_r")
          cbar = fig.colorbar(im1, ax=ax0)
          cbar.ax.get_yaxis().labelpad = 5
          cbar.ax.set_ylabel('A.U.', rotation=270)

      elif self.use_sources_Time:
          gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], height_ratios=[1, 3])
          ax0a = fig.add_subplot(gs_left[0])
          ax0b = fig.add_subplot(gs_left[1])

          ax0a.plot(self.sources.signals[0,:], label='pulse 1')
          ax0a.plot(self.sources.signals[1,:], label='pulse 2')
          ax0a.legend()
        
          xx = [point[0] for point in self.pulse_source_loc]
          yy = [point[1] for point in self.pulse_source_loc]
          plot_dim = self.size+self.pml_size+self.num_air_grid
          im1 = ax0b.imshow(jnp.zeros([plot_dim,plot_dim]), cmap="RdBu_r", vmin=0, vmax=1)
          cbar = fig.colorbar(im1, ax=ax0b)
          cbar.ax.get_yaxis().labelpad = 5
          cbar.ax.set_ylabel('A.U.', rotation=270)
          ax0b.scatter(xx,yy)
          ax0=ax0b
          del ax0b


      ax0.axis('off')
      ax0.set_title('Initial pressure')
      ax0.scatter(self.x, self.y, label="sensors", marker='.')
      ax0.legend(loc="upper right")

      ax1 = fig.add_subplot(gs[1])
      im1 = ax1.imshow(self.sound_speed.on_grid, cmap="RdBu_r")
      cbar = fig.colorbar(im1, ax=ax1)
      cbar.ax.get_yaxis().labelpad = 5
      cbar.ax.set_ylabel('A.U.', rotation=270)
      ax1.axis('off')
      ax1.set_title('Initial velocity field')
      ax1.scatter(self.x, self.y, label="sensors", marker='.')
      if save_me:
        if self.field_type == 'Graph':
          plt.savefig(self.path+f'/{self.field_type}_{self.graph_type}_p{self.p_vals}_eps{self.epsilon}_SummaryPlot.png')
        else:
          plt.savefig(self.path+f'/{self.field_type}_{self.graph_type}_var{self.cov_model_var}_scale{self.cov_model_len_scale}_SummaryPlot.png')
        plt.close()

      plt.tight_layout()
      plt.show()
    
    
    def plot_sensors_data(self,sensors_data,vmax=0.5,vmin=-0.5,scaling=0.1,save_me=False):
      ### Sensor Case!
      sensors_data = sensors_data.squeeze()
      _field = FourierSeries(sensors_data.T, self.domain)
      if isinstance(_field, Field):
              _field_arr = _field.on_grid
      #show_field(_field/scaling, "Recorded acoustic signals",vmax=vmax)
      plt.figure(figsize=(6,6), dpi=100)
      plt.imshow(_field_arr/scaling,aspect='auto',cmap='RdBu_r',vmax=vmax,vmin=vmin)
      plt.colorbar()
      plt.title("Waveforms Recorded at Sensors")
      plt.xlabel("Time step")
      plt.ylabel("Sensor position")
      plt.axis("on")
      self.sensor_data_array = _field_arr
      if save_me:
        if self.field_type == 'Graph':
          plt.savefig(self.path+f'/{self.field_type}_{self.graph_type}_p{self.p_vals}_eps{self.epsilon}_Waveform.png')
        else:
          plt.savefig(self.path+f'/{self.field_type}_{self.graph_type}_var{self.cov_model_var}_scale{self.cov_model_len_scale}_Waveform.png')
        plt.close()
      plt.show()


    def run_simulation(self, use_p0=True):
        pressure = self._compiled_simulator(use_p0)
        return pressure

    def _compiled_simulator(self,use_p0=True):
        medium = Medium(domain=self.domain,sound_speed=self.sound_speed,density=self.density_rho,pml_size=self.pml_size,attenuation=self.attentuation)
        if use_p0 == True :
          return simulate_wave_propagation(medium, self.time_axis, p0=self.p0)
        else:
          return simulate_wave_propagation(medium, self.time_axis, sources=self.sources)

    def compiled_simulator_sensors(self):
        ## called while using pressure pulse sources
        medium = Medium(domain=self.domain, sound_speed=self.sound_speed, density=self.density_rho, pml_size=self.pml_size, attenuation=self.attentuation)
        return simulate_wave_propagation(medium, self.time_axis, p0=self.p0, sensors=self.sensors)

    def compiled_simulator_sources(self):
        ## called while using gaussian time pulse sources
        medium = Medium(domain=self.domain, sound_speed=self.sound_speed, density=self.density_rho, pml_size=self.pml_size, attenuation=self.attentuation)
        return simulate_wave_propagation(medium, self.time_axis, sources=self.sources, sensors=self.sensors)

    def plot_pressure_field(self, pressure, t):
        show_field(pressure[t])
        plt.title(f"Pressure field at t={self.time_axis.to_array()[t]}")

    def make_gif(self, readpath = '../results/4fwd_model/', savepath='../results/4fwd_model/', dur=50):
      file_list = glob.glob(readpath)
      sorted_files = sorted(file_list, key=lambda x: int(re.findall(r'(\d+)', os.path.basename(x))[0]))
      frames = [Image.open(image) for image in sorted_files]
      frame_one = frames[0]
      frame_one.save(savepath, format="GIF", append_images=frames[1:],
                    save_all=True, duration=dur, loop=0)


## Take a look into the jit complie for all the four functions either together inside or keep them outside together.
## Check a run for each cases after this change. 
## Make sure that the sanity of the code is fine with all different cases.
# @jit
# def compiled_simulator_sources(domain,density_field,time_axis,pml_size=10,density_rho=1500.0,sources=None):
#     medium_v = Medium(domain=domain, sound_speed=density_field, density=density_rho, pml_size=pml_size)
#     print(medium_v)
#     return simulate_wave_propagation(medium_v, time_axis, sources=sources)

# @jit
# def compiled_simulator_sensor_sources(domain,density_field,time_axis,pml_size=10,density_rho=1500.0,sources=None,sensors=None):
#     medium_v = Medium(domain=domain, sound_speed=density_field, density=density_rho, pml_size=pml_size)
#     print(medium_v)
#     return simulate_wave_propagation(medium_v, time_axis, sources=sources, sensors=sensors)
