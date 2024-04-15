import './App.css';
import React from 'react';
import QuestionForm from './QuestionForm'; // Ensure this path is correct

const App = () => {
    return (

        <div className="app-container">
            <h1>Ask a Question</h1>
            <QuestionForm />
        </div>
    );
};

export default App; // Ensure you are exporting App
