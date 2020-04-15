import numpy as np
import folium
import folium.plugins
import matplotlib
import os

from lib.measures import (MeasureList, BetaMultiplierMeasure,
                      SocialDistancingForAllMeasure, BetaMultiplierMeasureByType,
                      SocialDistancingForPositiveMeasure, SocialDistancingByAgeMeasure, SocialDistancingForSmartTracing, ComplianceForAllMeasure)

class MapIllustrator():

    """
    Class to plot data on maps
    """

    def __init__(self):

        # map constants
        self.tile = 'OpenStreetMap'
        self.marker_radius = 3
        self.marker_opacity = 0.7
        self.marker_fill_opacity = 1
        self.heatmap_opacity = '.6'

        # colors
        self.intensity_color_map = matplotlib.cm.get_cmap('plasma')
        self.color_map = matplotlib.cm.get_cmap('tab10')
        self.gradient = {
            0.0: matplotlib.colors.rgb2hex(self.intensity_color_map(0)),
            0.2: matplotlib.colors.rgb2hex(self.intensity_color_map(0.1)),
            0.4: matplotlib.colors.rgb2hex(self.intensity_color_map(0.2)),
            0.6: matplotlib.colors.rgb2hex(self.intensity_color_map(0.4)),
            0.8: matplotlib.colors.rgb2hex(self.intensity_color_map(0.5)),
            1.0: matplotlib.colors.rgb2hex(self.intensity_color_map(1.0))
        }
        self.fill_layer_color = matplotlib.colors.rgb2hex(self.intensity_color_map(0.1))

    def _add_heatmap(self, map_obj, points, intensity=None, max_intensity=None):
        '''
        Creates a heatmap over a map depending on the density of the points given.
        '''

        if intensity is None:
            points_to_plot = list(zip([x[0] for x in points], [x[1] for x in points], [1]*len(points)))
            max_intensity=1
        else:
            points_to_plot = list(zip([x[0] for x in points], [x[1] for x in points], intensity))
            if max_intensity is None:
                max_intensity = max(intensity)

        folium.plugins.HeatMap(
            points_to_plot,
            min_opacity=0.3,
            max_val=max_intensity,
            gradient=self.gradient,
            radius=37, blur=70
        ).add_to(map_obj)

        # hacky way of changing heatmap opacity
        style_statement = '<style>.leaflet-heatmap-layer{opacity: ' + self.heatmap_opacity + '}</style>'
        map_obj.get_root().html.add_child(folium.Element(style_statement))

        return map_obj

    def _add_markers_with_category(self, map_obj, markers, categories, labels):
        '''
        Adds markers of different colors to a map, depending on their categories.
        '''

        for i_marker, marker in enumerate(markers):
            folium.CircleMarker(
                location = marker, 
                radius = self.marker_radius,
                color=matplotlib.colors.rgb2hex(self.color_map(categories[i_marker])),
                fill_color=matplotlib.colors.rgb2hex(self.color_map(categories[i_marker])),
                popup=labels[i_marker],
                opacity=self.marker_opacity,
                fill_opacity=self.marker_fill_opacity
            ).add_to(map_obj)

        return map_obj
    
    def __comp_checkins_in_a_day(self, sim, r, t):
        '''
        Computes the number of check-ins for all sites during a day starting at time t.
        '''

        site_checkins = np.zeros(sim.n_sites, dtype='bool')
        
        for site in range(sim.n_sites):
            for indiv in range(sim.n_people):
                if ( (not sim.measure_list[r].is_contained_prob(SocialDistancingForAllMeasure, t=t, j=indiv)) and
                     (not sim.measure_list[r].is_contained_prob(SocialDistancingForSmartTracing, t=t, j=indiv)) and
                     (not sim.measure_list[r].is_contained_prob(SocialDistancingByAgeMeasure, t=t, age=sim.people_age[r, indiv])) and
                     (not sim.measure_list[r].is_contained_prob(SocialDistancingForPositiveMeasure,
                                                                t=t, j=indiv,
                                                                state_posi_started_at=sim.state_started_at['posi'][r, :], 
                                                                state_posi_ended_at=sim.state_ended_at['posi'][r, :],
                                                                state_resi_started_at=sim.state_started_at['resi'][r, :],
                                                                state_dead_started_at=sim.state_started_at['dead'][r, :])) and
                     (not sim.state['dead'][r, indiv]) and
                    len(sim.mob[r].list_intervals_in_window_individual_at_site(indiv, site, t, t+24.0)) > 0):
                    site_checkins[site] += 1
        return site_checkins

    def population_map(self, bbox, map_name, home_loc):
        '''
        Visualizes the population on the town map with a heatmap.
        
        Parameters
        ----------
        bbox : (float, float, float, float)
            Coordinate bounding box
        map_name : string
            A name for the generated map
        home_loc : list of [float, float]
            List of home coordinates
        Returns
        -------
        m : MapIllustrator object
            The generated map
        '''

        # center map around the given bounding box
        center = ((bbox[0]+bbox[1])/2,(bbox[2]+bbox[3])/2)
        m = folium.Map(location=center,tiles=self.tile)
        m.fit_bounds([(bbox[0],bbox[2]),(bbox[1],bbox[3])])

        # generate heatmap of homes
        self._add_heatmap(m, home_loc)

        # save map as html
        if not os.path.exists('maps'):
            os.mkdir('maps')
        m.save('maps/'+map_name+'.html')

        return m

    def sites_map(self, bbox, site_loc, site_type, map_name, site_dict):
        '''
        Visualizes the given sites on the town map with markers of different
        color per site type.

        Parameters
        ----------
        bbox : (float, float, float, float)
            Coordinate bounding box
        site_loc : list of [float, float]
            List of site coordinates
        map_name : string
            A name for the generated map
        site_type : list of int
            List of site type
        site_dict : dictionary {int : string}
            Contains site types and their verbal interpretation
        
        Returns
        -------
        m : MapIllustrator object
            The generated map
        '''

        # center map around the given bounding box
        center = ((bbox[0]+bbox[1])/2,(bbox[2]+bbox[3])/2)
        m = folium.Map(location=center,tiles=self.tile)
        m.fit_bounds([(bbox[0],bbox[2]),(bbox[1],bbox[3])])

        # set marker labels as site types
        labels = [site_dict[site] for site in site_type]

        # add sites as markers
        self._add_markers_with_category(map_obj=m, markers=site_loc, categories=site_type, labels=labels)

        # save map as html
        if not os.path.exists('maps'):
            os.mkdir('maps')
        m.save('maps/'+map_name+'.html')

        return m

    def checkin_rate_map(self, bbox, site_loc, map_name, sim, t, max_checkin=None, r=0):
        '''
        Computes the rate of check-ins per site for a given day starting at time t
        and visualizes the rates with a heatmap. The map is saved as an html file.

        Parameters
        ----------
        bbox : (float, float, float, float)
            Coordinate bounding box
        site_loc : list of [float, float]
            List of site coordinates
        map_name : string
            A name for the generated map
        sim : DiseaseModel object
            Used for computing check-ins
        t : float
            Starting time of the target day
        max_checkin : int
            Relative check-in rate to determine the heatmap intensity
        Returns
        -------
        m : MapIllustrator object
            The generated map
        max_checkin : int
            Check-in rate of the site with most check-ins
        '''
        
        # center map around the given bounding box
        center = ((bbox[0]+bbox[1])/2,(bbox[2]+bbox[3])/2)
        m = folium.Map(location=center,tiles=self.tile)
        m.fit_bounds([(bbox[0],bbox[2]),(bbox[1],bbox[3])])

        # compute check-ins per site
        checkins = self.__comp_checkins_in_a_day(sim, r, t).tolist()
        
        if max_checkin is None:
            max_checkin = max(checkins)
        
        self._add_heatmap(map_obj=m, points=site_loc, intensity=checkins, max_intensity=max_checkin)

        # save map as html
        if not os.path.exists('maps'):
            os.mkdir('maps')
        m.save('maps/'+map_name+'.html')

        return m, max_checkin
    
