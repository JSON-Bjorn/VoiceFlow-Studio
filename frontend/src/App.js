import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Registration from './components/Registration';
import Login from './components/Login';
import Dashboard from './components/Dashboard';
import PodcastGeneration from './components/PodcastGeneration';
import Payment from './components/Payment';
import Admin from './components/Admin';
import ExampleLibrary from './components/ExampleLibrary';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/register" element={<Registration />} />
        <Route path="/login" element={<Login />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/generate" element={<PodcastGeneration />} />
        <Route path="/payment" element={<Payment />} />
        <Route path="/admin" element={<Admin />} />
        <Route path="/examples" element={<ExampleLibrary />} />
      </Routes>
    </Router>
  );
}

export default App;
