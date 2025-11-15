// src/App.jsx
import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import Papa from "papaparse";
import csv2020 from "./2020.csv";
import csv2021 from "./2021.csv";
import csv2022 from "./2022.csv";
import { MapPin, Cloud, TrendingUp, AlertTriangle } from "lucide-react";
import { Bar } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// --- constants ---
const MONSOON_WEIGHTS = [5,8,12,35,70,140,180,170,140,80,30,10];
const MONTHS_SHORT = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

const safeNum = (v) => {
  const s = String(v ?? "").replace(/[, ]+/g, "").trim();
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : NaN;
};
const levelToOneHot = (level) => {
  const L = String(level || '').trim().toUpperCase();
  return [L === 'H' ? 1 : 0, L === 'T' ? 1 : 0, L === 'D' ? 1 : 0];
};
function distributeAnnualToMonths(annual) {
  const total = MONSOON_WEIGHTS.reduce((a,b)=>a+b,0);
  return MONSOON_WEIGHTS.map((w,i)=>({
    month: MONTHS_SHORT[i],
    rainfall: annual * (w/total),
  }));
}

// --- parse CSV ---
async function parseFile(url, year) {
  return new Promise((resolve, reject) => {
    Papa.parse(url, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
        const rows = res.data.map((r) => ({
          year,
          level: String(r["District(D)/Taluk(T)/Hobli(H)"] || "").trim(),
          name: String(r["Name"] || "").trim(),
          normal: safeNum(r["Normal (mm)"]),
          actual: safeNum(r["Actual (mm)"]),
          dep: safeNum(r["%DEP"])
        })).filter(r => r.name && r.level);
        resolve(rows);
      },
      error: reject
    });
  });
}

// --- hierarchy ---
function buildHierarchy(rows) {
  const districts = [];
  const nodeHistory = new Map();
  const districtMap = new Map();

  let currentDistrict = null;
  let currentTaluk = null;

  rows.forEach(r => {
    const level = r.level.toUpperCase();
    const name = r.name;

    if(level === "D") {
      currentDistrict = name;
      currentTaluk = null;
      if(!districtMap.has(name)) {
        const district = { name, taluks: [] };
        districtMap.set(name, district);
        districts.push(district);
      }
      const key = `D|${name}`;
      nodeHistory.set(key, [...(nodeHistory.get(key)||[]), {normal: r.normal, dep: r.dep, actual: r.actual}]);
    } 
    else if(level === "T") {
      if(!currentDistrict) return;
      currentTaluk = name;
      const district = districtMap.get(currentDistrict);
      if(!district.taluks.find(t=>t.name===name)) district.taluks.push({name, hoblis: []});
      const key = `T|${name}|${currentDistrict}`;
      nodeHistory.set(key, [...(nodeHistory.get(key)||[]), {normal: r.normal, dep: r.dep, actual: r.actual}]);
    } 
    else if(level === "H") {
      if(!currentDistrict || !currentTaluk) return;
      const district = districtMap.get(currentDistrict);
      const taluk = district.taluks.find(t=>t.name===currentTaluk);
      if(!taluk.hoblis.includes(name)) taluk.hoblis.push(name);
      const key = `H|${name}|${currentDistrict}|${currentTaluk}`;
      nodeHistory.set(key, [...(nodeHistory.get(key)||[]), {normal: r.normal, dep: r.dep, actual: r.actual}]);
    }
  });

  return { districts, nodeHistory };
}

// --- train/load model ---
async function trainOrLoadModel(rows) {
  const saved = await tf.loadLayersModel('localstorage://rainfall-model').catch(()=>null);
  if(saved) return saved;

  const usable = rows.filter(r => Number.isFinite(r.normal) && Number.isFinite(r.actual));
  if(!usable.length) return null;

  const X = [], y = [];
  usable.forEach(r=>{
    const [h,t,d] = levelToOneHot(r.level);
    const yearScaled = (r.year-2020)/5;
    X.push([r.normal, isFinite(r.dep)?r.dep:0, yearScaled, h, t, d]);
    y.push(r.actual);
  });

  const xT = tf.tensor2d(X);
  const yT = tf.tensor2d(y, [y.length,1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units:16, activation:'relu', inputShape:[6] }));
  model.add(tf.layers.dense({ units:8, activation:'relu' }));
  model.add(tf.layers.dense({ units:1 }));
  model.compile({ optimizer: tf.train.adam(0.03), loss:'meanSquaredError' });

  await model.fit(xT, yT, { epochs: 20, batchSize:32, verbose:0, callbacks:{onEpochEnd:async()=>await tf.nextFrame()} });

  await model.save('localstorage://rainfall-model');
  xT.dispose(); yT.dispose();
  return model;
}

// --- prediction ---
function predictForNode(model, rowsForNode, level, targetYear = new Date().getFullYear()) {
  if(!rowsForNode.length) return null;
  const avgNormal = rowsForNode.map(r=>r.normal).filter(Number.isFinite).reduce((a,b)=>a+b,0)/rowsForNode.length;
  const avgDep = rowsForNode.map(r=>r.dep).filter(Number.isFinite).reduce((a,b)=>a+b,0)/rowsForNode.length || 0;
  const [h,t,d] = levelToOneHot(level);
  const x = tf.tensor2d([[avgNormal, avgDep, (targetYear-2020)/5, h, t, d]]);
  const y = model.predict(x);
  const val = y.dataSync()[0];
  x.dispose(); y.dispose();
  return Math.max(0, val);
}

// --- dynamic agriculture ---
function getCropRecommendations(annualRainfall) {
  if(annualRainfall < 500) return [
    { soilType: 'Red Soil', crops: 'Millets, Pulses', sowingPeriod: 'June-July', diseases: 'Leaf spot', precautions: 'Drought-tolerant varieties' },
    { soilType: 'Black Soil', crops: 'Sorghum, Wheat', sowingPeriod: 'June-July', diseases: 'Rust', precautions: 'Minimal irrigation' }
  ];
  if(annualRainfall < 1200) return [
    { soilType: 'Red Soil', crops: 'Groundnut, Cotton', sowingPeriod: 'June-July', diseases: 'Leaf spot, Root rot', precautions: 'Crop rotation' },
    { soilType: 'Black Soil', crops: 'Cotton, Sorghum, Wheat', sowingPeriod: 'June-July', diseases: 'Rust, Blight', precautions: 'Integrated pest management' }
  ];
  return [
    { soilType: 'Red Soil', crops: 'Paddy, Sugarcane', sowingPeriod: 'June-July', diseases: 'Blast, Root rot', precautions: 'Water management' },
    { soilType: 'Black Soil', crops: 'Paddy, Cotton', sowingPeriod: 'June-July', diseases: 'Blight', precautions: 'Ensure drainage' }
  ];
}

// --- main App ---
export default function App() {
  const [page, setPage] = useState("select");
  const [loading, setLoading] = useState(true);
  const [hier, setHier] = useState({ districts: [] });
  const [historyMap, setHistoryMap] = useState(new Map());
  const [model, setModel] = useState(null);
  const [district, setDistrict] = useState("");
  const [taluk, setTaluk] = useState("");
  const [hobli, setHobli] = useState("");
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    (async()=>{
      const [rows2020, rows2021, rows2022] = await Promise.all([
        parseFile(csv2020,2020), parseFile(csv2021,2021), parseFile(csv2022,2022)
      ]);
      const all = [...rows2020,...rows2021,...rows2022];
      const { districts, nodeHistory } = buildHierarchy(all);
      const m = await trainOrLoadModel(all);
      setHier({ districts });
      setHistoryMap(nodeHistory);
      setModel(m);
      setLoading(false);
    })();
  }, []);

  const districtOptions = hier.districts.map(d=>d.name);
  const talukOptions = district ? (hier.districts.find(d=>d.name===district)?.taluks.map(t=>t.name)||[]) : [];
  const hobliOptions = taluk ? (hier.districts.find(d=>d.name===district)?.taluks.find(t=>t.name===taluk)?.hoblis||[]) : [];

  function handlePredict() {
    const keyH = `H|${hobli}|${district}|${taluk}`;
    const keyT = `T|${taluk}|${district}`;
    const keyD = `D|${district}`;
    const rowsForH = historyMap.get(keyH) || [];
    const rowsForT = historyMap.get(keyT) || [];
    const rowsForD = historyMap.get(keyD) || [];
    const nodeRows = rowsForH.length ? rowsForH : (rowsForT.length ? rowsForT : rowsForD);
    const level = rowsForH.length ? "H" : rowsForT.length ? "T" : "D";
    const annual = predictForNode(model, nodeRows, level);
    setPrediction({
      location: `${district} > ${taluk} > ${hobli}`,
      annualPrediction: Math.round(annual),
      monthly: distributeAnnualToMonths(annual),
      crops: getCropRecommendations(Math.round(annual))
    });
    setPage("result");
  }

  if(loading) return <div className="p-6 text-center text-xl">Loading data and preparing model...</div>;

  // --- Attractive selection page ---
  if(page==="select") return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6 text-center">🌧️ Karnataka Rainfall Predictor</h1>
      <h2 className="text-xl font-semibold mb-4">Select District</h2>
      <div className="grid grid-cols-3 gap-4 mb-4">
        {districtOptions.map(d=>(
          <div key={d} className={`p-4 bg-blue-50 cursor-pointer hover:bg-blue-200 text-center rounded shadow`}
               onClick={()=>{setDistrict(d); setTaluk(""); setHobli("");}}>
            <MapPin className="mx-auto mb-2 text-blue-600"/>
            <p>{d}</p>
          </div>
        ))}
      </div>
      {district && <>
        <h2 className="text-xl font-semibold mb-2">Select Taluk</h2>
        <div className="grid grid-cols-3 gap-4 mb-4">
          {talukOptions.map(t=>(
            <div key={t} className={`p-3 bg-green-50 cursor-pointer hover:bg-green-200 text-center rounded shadow`}
                 onClick={()=>{setTaluk(t); setHobli("");}}>
              <p>{t}</p>
            </div>
          ))}
        </div>
      </>}
      {taluk && <>
        <h2 className="text-xl font-semibold mb-2">Select Hobli</h2>
        <div className="grid grid-cols-3 gap-4 mb-4">
          {hobliOptions.map(h=>(
            <div key={h} className={`p-3 bg-yellow-50 cursor-pointer hover:bg-yellow-200 text-center rounded shadow`}
                 onClick={()=>setHobli(h)}>
              <p>{h}</p>
            </div>
          ))}
        </div>
      </>}
      <button onClick={handlePredict} disabled={!district} className="mt-4 bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700">
        Predict Rainfall
      </button>
    </div>
  );

  // --- Result page with chart ---
  if(page==="result") return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4 flex items-center"><Cloud className="mr-2"/>Predicted Rainfall</h1>
      <p><strong>Location:</strong> {prediction.location}</p>
      <p><strong>Annual Rainfall:</strong> {prediction.annualPrediction} mm</p>
      <h2 className="mt-4 font-semibold">Monthly Distribution</h2>
      <Bar
        data={{
          labels: prediction.monthly.map(m=>m.month),
          datasets:[{ label:"Rainfall (mm)", data: prediction.monthly.map(m=>m.rainfall), backgroundColor:'rgba(54,162,235,0.6)' }]
        }}
        options={{ responsive:true, plugins:{ legend:{ display:false } } }}
      />
      <button onClick={()=>setPage("analysis")} className="mt-4 bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">Agriculture Analysis</button>
    </div>
  );

  // --- Dynamic agriculture analysis page ---
  if(page==="analysis") return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4 flex items-center"><TrendingUp className="mr-2"/>Agriculture Analysis</h1>
      {prediction.crops.map((r,i)=>(
        <div key={i} className="mb-4 p-4 border rounded bg-gray-50 shadow">
          <p><strong>Soil Type:</strong> {r.soilType}</p>
          <p><strong>Crops:</strong> {r.crops}</p>
          <p><strong>Sowing Period:</strong> {r.sowingPeriod}</p>
          <p><strong>Diseases:</strong> {r.diseases}</p>
          <p><strong>Precautions:</strong> {r.precautions}</p>
        </div>
      ))}
      <button onClick={()=>setPage("select")} className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">Back</button>
    </div>
  );
}