import React, { useState } from 'react';
import './ResponseDisplay.css';

const ResponseDisplay = ({ response }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="response-display">
      <p className={isExpanded ? "expanded" : ""}>
        {response}
      </p>
      <button onClick={() => setIsExpanded(!isExpanded)} className="toggle-button">
        {isExpanded ? 'Show less' : 'Show more'}
      </button>
    </div>
  );
};

export default ResponseDisplay;

