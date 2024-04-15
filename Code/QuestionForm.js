import React, { useState } from 'react';
import axios from 'axios';

const QuestionForm = () => {
    const [question, setQuestion] = useState('');
    const [response, setResponse] = useState('');
    const [links, setLinks] = useState([]);

    const handleQuestionChange = (event) => {
        setQuestion(event.target.value);
    };

    const fetchAnswer = async () => {
        try {
            const res = await axios.post('http://localhost:5000/greet', { question });
            setResponse(res.data.answer);
            setLinks(res.data.links);
        } catch (error) {
            console.error('Error fetching the answer:', error);
            setResponse('Failed to fetch answer. Please try again later.');
            setLinks([]);
        }
    };

    return (
        <div className="question-form">
            <textarea
                value={question}
                onChange={handleQuestionChange}
                placeholder="Ask your question..."
                rows="4"
                cols="50"
            />
            <button onClick={fetchAnswer}>Get Answer</button>
            <div className="response">{response}</div>
            <div></div>
            <div className="links">
                {links.map((link, index) => (
                    <div key={index}><a href={link} style={{ color: 'blue' }}>{link}</a></div>
                ))}
            </div>
        </div>
    );
};

export default QuestionForm;
