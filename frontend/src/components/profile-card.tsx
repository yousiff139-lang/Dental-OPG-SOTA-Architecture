"use client";

import { useState } from "react";

interface Developer {
    name: string;
    role: string;
    skills: string[];
    imageSrc: string;
    instagramUrl: string;
    githubUrl: string;
    email?: string;
}

const developers: Developer[] = [
    {
        name: "Karrar Al-Mayaly",
        role: "Full Stack Developer",
        skills: ["Python", "Java", "Node.js", "React", "TypeScript", "MySQL"],
        imageSrc: "/profiles/karrar.jpg",
        instagramUrl: "https://www.instagram.com/kode_pcs",
        githubUrl: "https://github.com/yousiff139-lang",
        email: "karrarmayaly@gmail.com",
    },
    {
        name: "Mohammed Majed",
        role: "Frontend Developer",
        skills: ["HTML", "CSS", "JavaScript"],
        imageSrc: "/profiles/mohammed.png",
        instagramUrl: "https://www.instagram.com/m0ho0/",
        githubUrl: "https://github.com/m0ho0",
    },
];

function Card3D({ dev }: { dev: Developer }) {
    return (
        <div className="card-parent">
            <div className="card-3d">
                {/* Animated Circle Logo */}
                <div className="card-logo">
                    <span className="circle circle1" />
                    <span className="circle circle2" />
                    <span className="circle circle3" />
                    <span className="circle circle4" />
                    <span className="circle circle5">
                        <img src={dev.imageSrc} alt={dev.name} className="circle-image" />
                    </span>
                </div>

                {/* Glass Effect */}
                <div className="card-glass" />

                {/* Content */}
                <div className="card-content">
                    <span className="card-title">{dev.name}</span>
                    <span className="card-role">{dev.role}</span>
                    <div className="card-skills">
                        {dev.skills.map((skill, i) => (
                            <span key={i} className="skill-tag">{skill}</span>
                        ))}
                    </div>
                </div>

                {/* Bottom Section */}
                <div className="card-bottom">
                    <div className="social-buttons">
                        {/* Instagram */}
                        <a href={dev.instagramUrl} target="_blank" rel="noopener noreferrer" className="social-btn">
                            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" />
                            </svg>
                        </a>

                        {/* GitHub */}
                        <a href={dev.githubUrl} target="_blank" rel="noopener noreferrer" className="social-btn">
                            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                            </svg>
                        </a>

                        {/* Gmail */}
                        {dev.email && (
                            <a href={`mailto:${dev.email}`} className="social-btn">
                                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-2.023 2.309-3.178 3.927-1.964L5.455 4.64 12 9.548l6.545-4.91 1.528-1.145C21.69 2.28 24 3.434 24 5.457z" />
                                </svg>
                            </a>
                        )}
                    </div>

                    <div className="view-profile">
                        <span>Developer</span>
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                            <path d="m6 9 6 6 6-6" />
                        </svg>
                    </div>
                </div>
            </div>
        </div>
    );
}

export function DeveloperProfiles() {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <>
            {/* Floating Developer Button */}
            <button
                className={`dev-widget-button ${isOpen ? "active" : ""}`}
                onClick={() => setIsOpen(!isOpen)}
                aria-label="Developer Info"
            >
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className="dev-icon">
                    <path d="M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0l4.6-4.6-4.6-4.6L16 6l6 6-6 6-1.4-1.4z" />
                </svg>
            </button>

            {/* Full Screen Popup */}
            {isOpen && (
                <div className="dev-fullscreen-overlay" onClick={() => setIsOpen(false)}>
                    <div className="dev-popup-3d" onClick={(e) => e.stopPropagation()}>
                        <div className="dev-popup-header">
                            <h2>Meet the Developers</h2>
                            <button className="close-btn" onClick={() => setIsOpen(false)}>
                                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
                                </svg>
                            </button>
                        </div>
                        <div className="dev-cards-container">
                            {developers.map((dev, index) => (
                                <Card3D key={index} dev={dev} />
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
