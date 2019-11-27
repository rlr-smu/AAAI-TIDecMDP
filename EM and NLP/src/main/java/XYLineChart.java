/**
 * Created by tarun on 11/6/17.
 */

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class XYLineChart {

    public static void genAndSaveGraph(String experiment, ArrayList<Double> objs, double nlpval, ArrayList<Double> times, double nlptime, double ovtime) {
        double[] yData = new double[objs.size()];
        double[] xData = new double[objs.size()];
        double[] timesyData = new double[objs.size()];
        double[] nlptimesyData = new double[objs.size()];
        double[] nlpyData = new double[objs.size()];

        double[] ovXdata = new double[1];
        ovXdata[0] = objs.size()-1;
        double[] ovYdata = new double[1];
        ovXdata[0] = ovtime;

        for (int i = 0; i < objs.size() ; i++) {
            xData[i] = i;
            yData[i] = objs.get(i);
            timesyData[i] = times.get(i);
        }
        Arrays.fill(nlpyData, nlpval);
        Arrays.fill(nlptimesyData, nlptime);

        // Create Chart
        XYChart chart = new XYChart(600, 400);
        chart.setTitle("Experiment: "+experiment);
        chart.setXAxisTitle("Iterations");
        chart.setYAxisTitle("Objective");
        XYSeries series = chart.addSeries("EM Objective", xData, yData);
        XYSeries series1 = chart.addSeries("NLP Objective", xData, nlpyData);
        series.setMarker(SeriesMarkers.NONE);
        series1.setMarker(SeriesMarkers.NONE);
        chart.getStyler().setPlotGridLinesVisible(false);
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideSE);

        XYChart timeschart = new XYChart(600, 400);
        timeschart.setTitle("Experiment: "+experiment);
        timeschart.setXAxisTitle("Iterations");
        timeschart.setYAxisTitle("Time(in seconds)");
        XYSeries series2 = timeschart.addSeries("EM Time", xData, timesyData);
        XYSeries series3 = timeschart.addSeries("NLP Time", xData, nlptimesyData);
        //XYSeries series4 = timeschart.addSeries("Overall EM Time", ovXdata, ovYdata);
        series2.setMarker(SeriesMarkers.NONE);
        series3.setMarker(SeriesMarkers.NONE);
        //series4.setMarker(SeriesMarkers.DIAMOND);
        timeschart.getStyler().setPlotGridLinesVisible(false);
        timeschart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);


        List<XYChart> charts = new ArrayList<XYChart>();
        charts.add(chart);
        charts.add(timeschart);

        //new SwingWrapper<XYChart>(charts).displayChartMatrix();
        try {
            BitmapEncoder.saveBitmapWithDPI(chart, Config.workDir+"Results/Objective_"+experiment, BitmapEncoder.BitmapFormat.PNG, 300);
            BitmapEncoder.saveBitmapWithDPI(timeschart, Config.workDir+"Results/Time_"+experiment, BitmapEncoder.BitmapFormat.PNG, 300);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}