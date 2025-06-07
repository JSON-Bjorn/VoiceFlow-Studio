import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "VoiceFlow Studio - Create Professional AI Podcasts",
  description: "Transform any topic into engaging, professional-quality podcasts with our AI-powered multi-agent system. Create studio-quality podcasts in minutes.",
  keywords: ["AI podcasts", "podcast creation", "artificial intelligence", "voice generation", "audio content"],
  authors: [{ name: "VoiceFlow Studio" }],
  creator: "VoiceFlow Studio",
  publisher: "VoiceFlow Studio",
  openGraph: {
    title: "VoiceFlow Studio - Create Professional AI Podcasts",
    description: "Transform any topic into engaging, professional-quality podcasts with our AI-powered multi-agent system.",
    siteName: "VoiceFlow Studio",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "VoiceFlow Studio - Create Professional AI Podcasts",
    description: "Transform any topic into engaging, professional-quality podcasts with our AI-powered multi-agent system.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
