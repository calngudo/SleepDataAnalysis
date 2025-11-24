import React, { useState } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Moon, TrendingUp, Heart, Activity, Coffee, Smartphone, Clock, ChevronRight, Plus, Calendar } from 'lucide-react';

const SleepTrackerApp = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showAddData, setShowAddData] = useState(false);

  // Sample data for the past week
  const sleepData = [
    { day: 'Mon', duration: 7.5, quality: 82, deepSleep: 22, remSleep: 24, lightSleep: 54, steps: 8500, hrv: 65, bedtime: '22:30', wakeup: '06:00', screenTime: 45, caffeine: '14:00' },
    { day: 'Tue', duration: 6.8, quality: 74, deepSleep: 18, remSleep: 22, lightSleep: 60, steps: 6200, hrv: 58, bedtime: '23:15', wakeup: '06:00', screenTime: 90, caffeine: '16:30' },
    { day: 'Wed', duration: 8.2, quality: 88, deepSleep: 25, remSleep: 26, lightSleep: 49, steps: 9100, hrv: 72, bedtime: '22:00', wakeup: '06:15', screenTime: 30, caffeine: '13:00' },
    { day: 'Thu', duration: 7.0, quality: 79, deepSleep: 20, remSleep: 23, lightSleep: 57, steps: 7800, hrv: 63, bedtime: '22:45', wakeup: '05:45', screenTime: 60, caffeine: '15:00' },
    { day: 'Fri', duration: 6.5, quality: 71, deepSleep: 17, remSleep: 21, lightSleep: 62, steps: 5900, hrv: 56, bedtime: '00:00', wakeup: '06:30', screenTime: 120, caffeine: '17:00' },
    { day: 'Sat', duration: 8.5, quality: 91, deepSleep: 27, remSleep: 28, lightSleep: 45, steps: 10200, hrv: 78, bedtime: '22:15', wakeup: '06:45', screenTime: 20, caffeine: '12:00' },
    { day: 'Sun', duration: 7.8, quality: 85, deepSleep: 23, remSleep: 25, lightSleep: 52, steps: 8900, hrv: 68, bedtime: '22:30', wakeup: '06:15', screenTime: 40, caffeine: '13:30' }
  ];

  const avgSleepDuration = (sleepData.reduce((sum, day) => sum + day.duration, 0) / sleepData.length).toFixed(1);
  const avgQuality = Math.round(sleepData.reduce((sum, day) => sum + day.quality, 0) / sleepData.length);
  const avgHRV = Math.round(sleepData.reduce((sum, day) => sum + day.hrv, 0) / sleepData.length);

  const insights = [
    {
      title: "Your Sleep Sweet Spot",
      description: "You sleep 45 minutes longer and achieve 12% better quality when you stop screen time by 9 PM",
      impact: "high",
      icon: Smartphone
    },
    {
      title: "Activity Boost",
      description: "Days with 8,000+ steps correlate with 15% more deep sleep. Your body loves movement!",
      impact: "high",
      icon: Activity
    },
    {
      title: "Caffeine Timing",
      description: "Having caffeine after 2 PM reduces your REM sleep by an average of 8%. Try cutting off earlier.",
      impact: "medium",
      icon: Coffee
    },
    {
      title: "Consistency Matters",
      description: "Your best sleep happens when you're in bed by 10:30 PM. Weekend sleep-ins disrupt your rhythm.",
      impact: "medium",
      icon: Clock
    }
  ];

  const recommendations = [
    "Start your wind-down routine at 9:30 PM with dim lighting",
    "Try the 4-7-8 breathing technique before bed tonight",
    "Aim for 8,000 steps today - it helps you sleep deeper",
    "Keep your bedroom between 65-68°F for optimal sleep",
    "Consider a 10-minute meditation session this evening"
  ];

  const Dashboard = () => (
    <div className="space-y-6">
      {/* Sleep Score */}
      <div className="bg-gradient-to-br from-blue-500 to-cyan-600 rounded-2xl p-6 text-white">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-sm opacity-90 mb-1">Your Sleep Score</div>
            <div className="text-5xl font-bold">{avgQuality}</div>
          </div>
          <div className="bg-white bg-opacity-20 rounded-full p-4">
            <Moon className="w-10 h-10" />
          </div>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <TrendingUp className="w-4 h-4" />
          <span>+5 points from last week</span>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <div className="text-gray-600 text-xs mb-1">Avg Duration</div>
          <div className="text-2xl font-bold text-gray-800">{avgSleepDuration}h</div>
          <div className="text-xs text-green-600 mt-1">Target: 7-9h</div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <div className="text-gray-600 text-xs mb-1">Avg HRV</div>
          <div className="text-2xl font-bold text-gray-800">{avgHRV}ms</div>
          <div className="text-xs text-blue-600 mt-1">Good range</div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <div className="text-gray-600 text-xs mb-1">Deep Sleep</div>
          <div className="text-2xl font-bold text-gray-800">22%</div>
          <div className="text-xs text-orange-600 mt-1">Aim for 25%</div>
        </div>
      </div>

      {/* Sleep Duration Chart */}
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Sleep Duration Trend</h3>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={sleepData}>
            <defs>
              <linearGradient id="colorDuration" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="day" stroke="#999" style={{ fontSize: '12px' }} />
            <YAxis stroke="#999" style={{ fontSize: '12px' }} domain={[0, 10]} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
              formatter={(value) => `${value}h`}
            />
            <Area type="monotone" dataKey="duration" stroke="#3b82f6" strokeWidth={2} fill="url(#colorDuration)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Sleep Quality Breakdown */}
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Sleep Quality Breakdown</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={sleepData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="day" stroke="#999" style={{ fontSize: '12px' }} />
            <YAxis stroke="#999" style={{ fontSize: '12px' }} />
            <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }} />
            <Bar dataKey="deepSleep" stackId="a" fill="#1e40af" radius={[0, 0, 0, 0]} />
            <Bar dataKey="remSleep" stackId="a" fill="#3b82f6" radius={[0, 0, 0, 0]} />
            <Bar dataKey="lightSleep" stackId="a" fill="#93c5fd" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div className="flex gap-4 mt-4 justify-center text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-900 rounded"></div>
            <span className="text-gray-600">Deep</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded"></div>
            <span className="text-gray-600">REM</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-300 rounded"></div>
            <span className="text-gray-600">Light</span>
          </div>
        </div>
      </div>

      {/* Quick Tip */}
      <div className="bg-gradient-to-r from-cyan-50 to-blue-50 rounded-xl p-5 border border-cyan-200">
        <div className="flex items-start gap-3">
          <div className="bg-cyan-500 rounded-full p-2 mt-1">
            <Moon className="w-4 h-4 text-white" />
          </div>
          <div>
            <div className="font-semibold text-gray-800 mb-1">Tonight's Tip</div>
            <div className="text-sm text-gray-700">Try reading for 15 minutes instead of scrolling. Your data shows this helps you fall asleep 20 minutes faster.</div>
          </div>
        </div>
      </div>
    </div>
  );

  const InsightsTab = () => (
    <div className="space-y-4">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">AI Insights</h2>
        <p className="text-gray-600 text-sm">Personalized patterns detected from your sleep data</p>
      </div>

      {insights.map((insight, idx) => (
        <div key={idx} className="bg-white rounded-xl p-5 border border-gray-200 hover:shadow-md transition-shadow">
          <div className="flex items-start gap-4">
            <div className={`rounded-full p-3 ${insight.impact === 'high' ? 'bg-orange-100' : 'bg-blue-100'}`}>
              <insight.icon className={`w-5 h-5 ${insight.impact === 'high' ? 'text-orange-600' : 'text-blue-600'}`} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <h3 className="font-semibold text-gray-800">{insight.title}</h3>
                <span className={`text-xs px-2 py-1 rounded-full ${insight.impact === 'high' ? 'bg-orange-100 text-orange-700' : 'bg-blue-100 text-blue-700'}`}>
                  {insight.impact === 'high' ? 'High Impact' : 'Medium Impact'}
                </span>
              </div>
              <p className="text-sm text-gray-600">{insight.description}</p>
            </div>
            <ChevronRight className="w-5 h-5 text-gray-400 mt-1" />
          </div>
        </div>
      ))}

      <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border border-green-200 mt-6">
        <h3 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
          <Heart className="w-5 h-5 text-green-600" />
          Recommended Actions
        </h3>
        <div className="space-y-2">
          {recommendations.map((rec, idx) => (
            <div key={idx} className="flex items-start gap-3">
              <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-white text-xs">{idx + 1}</span>
              </div>
              <p className="text-sm text-gray-700">{rec}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const AddDataForm = () => (
    <div className="bg-white rounded-xl p-6 border border-gray-200">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Log Sleep Data</h3>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm text-gray-600 mb-1 block">Bedtime</label>
            <input type="time" className="w-full px-3 py-2 border border-gray-300 rounded-lg" defaultValue="22:30" />
          </div>
          <div>
            <label className="text-sm text-gray-600 mb-1 block">Wake Time</label>
            <input type="time" className="w-full px-3 py-2 border border-gray-300 rounded-lg" defaultValue="06:30" />
          </div>
        </div>
        
        <div>
          <label className="text-sm text-gray-600 mb-1 block">Sleep Quality (1-100)</label>
          <input type="number" className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="85" />
        </div>

        <div>
          <label className="text-sm text-gray-600 mb-1 block">Daily Steps</label>
          <input type="number" className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="8000" />
        </div>

        <div>
          <label className="text-sm text-gray-600 mb-1 block">Screen Time Before Bed (minutes)</label>
          <input type="number" className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="30" />
        </div>

        <div>
          <label className="text-sm text-gray-600 mb-1 block">Last Caffeine Time</label>
          <input type="time" className="w-full px-3 py-2 border border-gray-300 rounded-lg" defaultValue="14:00" />
        </div>

        <div className="flex gap-3 pt-2">
          <button className="flex-1 bg-blue-500 text-white py-2.5 rounded-lg font-medium hover:bg-blue-600 transition-colors">
            Save Entry
          </button>
          <button 
            onClick={() => setShowAddData(false)}
            className="px-6 bg-gray-200 text-gray-700 py-2.5 rounded-lg font-medium hover:bg-gray-300 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>

      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="text-sm text-gray-600 mb-3">Or connect your device:</div>
        <div className="grid grid-cols-2 gap-3">
          <button className="py-3 px-4 border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all flex items-center justify-center gap-2 text-sm font-medium text-gray-700">
            <Activity className="w-4 h-4" />
            Apple Health
          </button>
          <button className="py-3 px-4 border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all flex items-center justify-center gap-2 text-sm font-medium text-gray-700">
            <Heart className="w-4 h-4" />
            Google Fit
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl p-2">
                <Moon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-800">Sleep Insights</h1>
                <p className="text-xs text-gray-600">Your personal sleep coach</p>
              </div>
            </div>
            <button 
              onClick={() => setShowAddData(!showAddData)}
              className="bg-blue-500 text-white px-4 py-2 rounded-lg flex items-center gap-2 hover:bg-blue-600 transition-colors text-sm font-medium"
            >
              <Plus className="w-4 h-4" />
              Add Data
            </button>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4">
          <div className="flex gap-1">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`px-6 py-3 text-sm font-medium transition-colors relative ${
                activeTab === 'dashboard'
                  ? 'text-blue-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Dashboard
              {activeTab === 'dashboard' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600"></div>
              )}
            </button>
            <button
              onClick={() => setActiveTab('insights')}
              className={`px-6 py-3 text-sm font-medium transition-colors relative ${
                activeTab === 'insights'
                  ? 'text-blue-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              AI Insights
              {activeTab === 'insights' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600"></div>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 py-6">
        {showAddData && <AddDataForm />}
        {!showAddData && (activeTab === 'dashboard' ? <Dashboard /> : <InsightsTab />)}
      </div>
    </div>
  );
};

export default SleepTrackerApp;