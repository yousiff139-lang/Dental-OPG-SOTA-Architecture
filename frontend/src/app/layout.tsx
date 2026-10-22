import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";
import "./globals.css";
import Provider from "@/components/provider";
import { DeveloperProfiles } from "@/components/profile-card";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  weight: "variable",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Dental X-Ray AI Analysis",
  description: "AI-powered dental X-ray analysis for cavity and periapical lesion detection",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${spaceGrotesk.variable} antialiased min-h-screen font-sans bg-background text-foreground`}
        suppressHydrationWarning
      >
        <Provider>
          {children}
          <DeveloperProfiles />
        </Provider>
      </body>
    </html>
  );
}
