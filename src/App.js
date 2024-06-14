import React, { useState } from 'react';
import Header from './components/Header';
import QueryForm from './components/QueryForm';
import ResponseDisplay from './components/ResponseDisplay';
import './App.css';

const App = () => {
  const [response, setResponse] = useState('');

  return (
    <div className="App">
      <Header />
      <main>
        <div className="welcome-section">
          <img src="https://thechurchco-production.s3.amazonaws.com/uploads/sites/4890/2022/02/larry-kim-headshot-web.jpg" alt="Pastor Headshot" className="pastor-headshot" />
          <h2>Welcome to CSC's AI Spiritual Assistant!</h2>
        </div>
        <QueryForm setResponse={setResponse} />
        <ResponseDisplay response={response} />
      </main>
      <footer>
        <p>Powered by SolaceAI</p>
      </footer>
    </div>
  );
};

export default App;
