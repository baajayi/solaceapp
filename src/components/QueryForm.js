import React, { useState } from 'react';
import axios from 'axios';
import './QueryForm.css';

const QueryForm = ({ setResponse }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('https://solacecencov.onrender.com/ask', { query });
      setResponse(response.data.response);
    } catch (error) {
      setResponse('Error: Unable to retrieve response');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="query-form">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a question"
        required
      />
      <button type="submit">Send</button>
    </form>
  );
};

export default QueryForm;
